import assert from "node:assert/strict";
import test from "node:test";
import OpenAI from "openai";
import { CaptureRuntime } from "../dist/capture.js";
import { setCaptureRuntime, wrap } from "../dist/wrap.js";

function capture(t) {
  const rows=[];
  setCaptureRuntime(new CaptureRuntime({enqueue(row){rows.push(row);return true;}},
    {captureText:false,appRoot:"",skipFrames:[],textMaxBytes:1024}));
  t.after(()=>setCaptureRuntime());
  return rows;
}

test("real Responses client preserves served tier separately from requested tier",async(t)=>{
  const rows=capture(t);
  const client=wrap(new OpenAI({apiKey:"test",maxRetries:0,fetch:async(_url,init)=>{
    assert.equal(JSON.parse(init.body).service_tier,"flex");
    return Response.json({id:"resp_test",object:"response",status:"completed",service_tier:"default",model:"example-model",output:[],usage:{input_tokens:10,output_tokens:2}},{headers:{"x-request-id":"req_test"}});
  }}),"openai");
  const result=await client.responses.create({model:"example-model",input:"Synthetic",service_tier:"flex"}).withResponse();
  assert.equal(result.data.service_tier,"default");
  assert.equal(result.response.headers.get("x-request-id"),"req_test");
  assert.equal(rows.length,1);
  assert.equal(rows[0].service_tier,"default");
  assert.equal(rows[0].requested_service_tier,"flex");
  assert.equal(rows[0].content_opted_in,false);
});

test("a requested tier does not establish the served tier",async(t)=>{
  const rows=capture(t);
  const client=wrap({responses:{create:async()=>({id:"resp_test",status:"completed"})}},"openai");
  await client.responses.create({model:"example-model",input:"Synthetic",service_tier:"flex"});
  assert.equal(rows[0].requested_service_tier,"flex");
  assert.equal(rows[0].service_tier,undefined);
});

for(const [name,path,chunks] of [
  ["chat stream","chat",[{service_tier:"flex",choices:[{delta:{content:"OK"}}]},{choices:[],usage:{prompt_tokens:1,completion_tokens:1}}]],
  ["Responses stream","responses",[{type:"response.created",response:{service_tier:"flex"}},{type:"response.completed",response:{service_tier:"default",status:"completed"}},{type:"response.done"}]],
])test(`${name} retains the latest explicit tier across trailing chunks`,async(t)=>{
  const rows=capture(t);
  const method=async()=>({async *[Symbol.asyncIterator](){yield* chunks;}});
  const client=wrap(path==="chat"?{chat:{completions:{create:method}}}:{responses:{create:method}},"openai");
  const stream=path==="chat"?await client.chat.completions.create({model:"example-model",messages:[],stream:true,service_tier:"flex"}):await client.responses.create({model:"example-model",input:"Synthetic",stream:true,service_tier:"flex"});
  for await(const chunk of stream)assert.ok(chunks.includes(chunk));
  assert.equal(rows.length,1);
  assert.equal(rows[0].service_tier,path==="chat"?"flex":"default");
  assert.equal(rows[0].requested_service_tier,"flex");
});

test("malformed tiers are omitted rather than inferred",async(t)=>{
  const rows=capture(t);
  const client=wrap({chat:{completions:{create:async()=>({service_tier:{unexpected:true},choices:[]})}}},"openai");
  await client.chat.completions.create({model:"example-model",messages:[],service_tier:""});
  assert.equal(rows[0].service_tier,undefined);
  assert.equal(rows[0].requested_service_tier,undefined);
});

const tierCases = [
  ["Anthropic nested usage", "anthropic", "messages.create", {service_tier:"auto"}, {usage:{service_tier:"standard"}}, "standard", "auto"],
  ["Google Vertex Flex", "google", "models.generate_content", {config:{httpOptions:{headers:{"X-Vertex-AI-LLM-Shared-Request-Type":"flex"}}}}, {usageMetadata:{trafficType:"ON_DEMAND_FLEX"}}, "flex", "flex"],
  ["Google standard fallback", "google", "models.generate_content", {config:{serviceTier:"flex"}}, {usageMetadata:{trafficType:"ON_DEMAND"}}, "standard", "flex"],
  ["Google snake case", "google", "models.generate_content", {config:{service_tier:"flex"}}, {usage_metadata:{traffic_type:"ON_DEMAND_FLEX"}}, "flex", "flex"],
  ["Google Developer tier", "google", "models.generate_content", {config:{serviceTier:"flex"}}, {usageMetadata:{serviceTier:"flex"}}, "flex", "flex"],
  ["OpenAI batch transport", "openai", "batch.responses", {batch:true}, {service_tier:"default"}, "batch", undefined],
  ["Anthropic batch transport", "anthropic", "batch.messages", {batch:true}, {usage:{service_tier:"standard"}}, "batch", undefined],
  ["requested batch alone", "anthropic", "messages.create", {service_tier:"batch"}, {}, undefined, "batch"],
];
function finishRow(t,provider,endpoint,request,response,extra={}) {
  const rows=capture(t);
  const runtime=new CaptureRuntime({enqueue(row){rows.push(row);return true;}},{captureText:false,appRoot:"",skipFrames:[],textMaxBytes:1024});
  runtime.finish(runtime.start(provider,endpoint,{model:"example-model",...request}),response,extra);
  return rows[0];
}
for (const [name,provider,endpoint,request,response,served,requested] of tierCases) {
  test(name, (t) => {
    const row=finishRow(t,provider,endpoint,request,response);
    assert.equal(row.service_tier,served);
    assert.equal(row.requested_service_tier,requested);
  });
}
for (const [provider,endpoint,chunks] of [
  ["anthropic","messages.create",[{type:"message_start",message:{usage:{service_tier:"standard"}}},{type:"message_stop"}]],
  ["google","models.generate_content.stream",[{usageMetadata:{trafficType:"ON_DEMAND_FLEX"}},{candidates:[]}]],
]) test(`${provider} stream retains tier before a trailing event`, (t) => {
  const row=finishRow(t,provider,endpoint,{},chunks.at(-1),{stream:true,responseChunks:chunks});
  assert.equal(row.service_tier,provider==="google"?"flex":"standard");
});
test("failed batch results do not establish a served tier", (t) => {
  const row=finishRow(t,"anthropic","batch.messages",{batch:true,service_tier:"batch"},{},{status:"error"});
  assert.equal(row.service_tier,undefined);
});

test("batch adapters do not invent a requested service tier", async (t) => {
  const rows=capture(t);
  const item={custom_id:"tier-regression",result:{type:"succeeded",message:{model:"example-model",usage:{service_tier:"standard"}}}};
  const client=wrap({messages:{batches:{results:async()=>({async *[Symbol.asyncIterator](){yield item;}})}}},"anthropic");
  for await (const value of await client.messages.batches.results("batch-tier-regression")) assert.equal(value,item);
  assert.equal(rows[0].requested_service_tier,undefined);
  assert.equal(rows[0].service_tier,"batch");
});

test("OpenAI batch capture does not invent a requested service tier", async (t) => {
  const rows=capture(t);
  const item={id:"tier-regression",custom_id:"tier-regression",response:{status_code:200,body:{model:"example-model",object:"response",service_tier:"default"}}};
  const text=JSON.stringify(item);
  const client=wrap({files:{content:async()=>({text:async()=>text})}},"openai");
  assert.equal(await (await client.files.content("file-tier-regression")).text(),text);
  assert.equal(rows[0].requested_service_tier,undefined);
  assert.equal(rows[0].service_tier,"batch");
});

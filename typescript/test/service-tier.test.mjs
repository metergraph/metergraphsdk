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

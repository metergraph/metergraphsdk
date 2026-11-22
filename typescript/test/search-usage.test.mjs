import assert from 'node:assert/strict';
import test from 'node:test';
import { CaptureRuntime } from '../dist/capture.js';

const search = id => ({type:'web_search_call',id,status:'completed'});
function record(response, extra={}, request={}) {
  const rows=[];
  const runtime=new CaptureRuntime({enqueue(row){rows.push(row);return true;}},
    {captureText:false,appRoot:'',skipFrames:[],textMaxBytes:32});
  runtime.finish(runtime.start('openai','responses.create',{model:'example-model',...request}),response,extra);
  return rows[0];
}
test('counts executed searches without retaining text or counting declared tools',()=>{
  const response={output:[search('ws_1'),{type:'function_call',name:'lookup'},search('ws_2'),search('ws_3')]};
  const row=record(response,{}, {tools:[{type:'web_search'}]});
  assert.equal(row.web_search_calls,3);
  assert.equal(row.content_opted_in,false);
  assert.equal(record({output:[]},{},{tools:[{type:'web_search'}]}).web_search_calls,0);
  assert.equal(record({choices:[]}).web_search_calls,undefined);
});
test('stream snapshots and repeated item events do not double count searches',()=>{
  const output=[search('ws_1'),search('ws_2')];
  const chunks=[
    {type:'response.output_item.done',item:output[0],output_index:0},
    {type:'response.output_item.done',item:output[0],output_index:0},
    {type:'response.completed',response:{output}},
    {type:'response.done'},
  ];
  assert.equal(record(chunks.at(-1),{stream:true,responseChunks:chunks}).web_search_calls,2);
  assert.equal(record(chunks[1],{stream:true,responseChunks:chunks.slice(0,2)}).web_search_calls,1);
});

test('wrapped native OpenAI client preserves the result and records searches',async(t)=>{
  const {default:OpenAI}=await import('openai');
  const {wrap,setCaptureRuntime}=await import('../dist/wrap.js');
  const rows=[];
  setCaptureRuntime(new CaptureRuntime({enqueue(row){rows.push(row);return true;}},
    {captureText:false,appRoot:'',skipFrames:[],textMaxBytes:32}));
  t.after(()=>setCaptureRuntime());
  const body={id:'resp_search',object:'response',model:'example-model',status:'completed',
    output:[search('ws_1'),search('ws_2'),search('ws_3')],usage:{input_tokens:100,output_tokens:20}};
  let calls=0;
  const client=wrap(new OpenAI({apiKey:'test',maxRetries:0,fetch:async()=>{
    calls++;return Response.json(body,{headers:{'x-request-id':'req_search'}});
  }}),'openai');
  const result=await client.responses.create({model:'example-model',input:'Synthetic'});
  assert.deepEqual(result,{...body,output_text:''});
  assert.equal(calls,1);
  assert.equal(rows.length,1);
  assert.equal(rows[0].request_id,'req_search');
  assert.equal(rows[0].web_search_calls,3);
});

function providerRecord(provider, response, chunks=[], request={}) {
  const rows=[];
  const runtime=new CaptureRuntime({enqueue(row){rows.push(row);return true;}},
    {captureText:false,appRoot:'',skipFrames:[],textMaxBytes:32});
  runtime.finish(runtime.start(provider,'responses.create',{model:'example-model',...request}),response,{responseChunks:chunks});
  return rows[0];
}
test('Anthropic retains latest explicit search/fetch usage, including zero',()=>{
  const chunks=[{type:'message_start',message:{usage:{server_tool_use:{web_search_requests:0}}}},
    {type:'message_delta',usage:{server_tool_use:{web_search_requests:2,web_fetch_requests:1}}}];
  assert.deepEqual(providerRecord('anthropic',{},chunks).server_tool_use,{web_search_requests:2,web_fetch_requests:1});
  assert.deepEqual(providerRecord('anthropic',{usage:{server_tool_use:{web_search_requests:0}}}).server_tool_use,{web_search_requests:0});
});
test('Google deduplicates nonempty grounding queries across stream snapshots',()=>{
  const a={candidates:[{groundingMetadata:{webSearchQueries:['one','','one']}}]};
  const b={candidates:[{grounding_metadata:{web_search_queries:['one','two',' ']}}]};
  const row=providerRecord('google',{},[a,b]);
  assert.equal(row.grounding_queries,2);assert.equal(row.grounded_prompt,1);
  assert.equal(providerRecord('google',{}).grounding_queries,undefined);
});
test('Perplexity uses invocation usage before output fallback and preserves API/type',()=>{
  const request={tools:[{type:'web_search',search_type:'fast'}]};
  const response={output:[{type:'search_results'}],usage:{tool_calls_details:{search_web:{invocation:3}}}};
  const row=providerRecord('perplexity',response,[],request);
  assert.equal(row.web_search_calls,3);assert.equal(row.retrieval_api,'agent');assert.equal(row.search_type,'fast');
  assert.equal(providerRecord('perplexity',{output:[{type:'search_results'},{type:'search_results'}]}).web_search_calls,2);
  assert.equal(providerRecord('perplexity',{output:[],usage:{tool_calls_details:{search_web:{invocation:0}}}}).web_search_calls,0);
  assert.equal(providerRecord('perplexity',{}).web_search_calls,undefined);
});

test('Perplexity OpenAI-compatible client keeps provider identity',async(t)=>{
  const {default:OpenAI}=await import('openai');
  const {wrap,setCaptureRuntime}=await import('../dist/wrap.js');
  const rows=[];
  setCaptureRuntime(new CaptureRuntime({enqueue(row){rows.push(row);return true;}},
    {captureText:false,appRoot:'',skipFrames:[],textMaxBytes:32}));
  t.after(()=>setCaptureRuntime());
  const body={id:'resp_search',object:'response',model:'perplexity/sonar',status:'completed',output:[{type:'search_results'}]};
  const client=wrap(new OpenAI({apiKey:'test',baseURL:'https://api.perplexity.ai',maxRetries:0,fetch:async()=>Response.json(body)}));
  await client.responses.create({model:'perplexity/sonar',input:'Synthetic',tools:[{type:'web_search',search_type:'web'}]});
  assert.equal(rows.length,1);assert.equal(rows[0].provider,'perplexity');assert.equal(rows[0].web_search_calls,1);
});

test('Perplexity resolves the bare web-search default without guessing managed configurations',()=>{
  const response={output:[{type:'search_results'}]};
  const record=request=>providerRecord('perplexity',response,[],request);
  assert.equal(record({tools:[{type:'web_search'}]}).search_type,'web');
  assert.equal(record({tools:[{type:'web_search'}],reasoning:{effort:'high'}}).search_type,'web');
  for (const extra of [{effort:'ultra'},{effort:'low@2026-10-16.0'},{effort:'unknown'},{profile:'example-profile'}]) {
    assert.equal(record(extra).search_type,undefined);
    assert.equal(record({tools:[{type:'web_search'}],...extra}).search_type,undefined);
    assert.equal(record({tools:[{type:'web_search',search_type:'web'}],...extra}).search_type,'web');
    assert.equal(record({tools:[{type:'web_search',search_type:'fast'}],...extra}).search_type,'fast');
  }
  for (const tools of [undefined,[],[{type:'web_search',search_type:null}],[{type:'web_search',search_type:'unknown'}],[{type:'web_search'},{type:'web_search',search_type:'fast'}]]) {
    assert.equal(record({tools}).search_type,undefined);
  }
});

test('Perplexity default does not change other provider usage',async()=>{
  const {retrievalUsage}=await import('../dist/retrieval.js');
  const request={tools:[{type:'web_search'}]};
  const response={output:[{type:'web_search_call'},{type:'search_results'}],
    usage:{server_tool_use:{web_search_requests:2}},
    candidates:[{groundingMetadata:{webSearchQueries:['synthetic']}}]};
  for(const [provider,expected] of [
    ['openai',{web_search_calls:1}],
    ['anthropic',{server_tool_use:{web_search_requests:2}}],
    ['google',{grounding_queries:1,grounded_prompt:1}],
    ['deepseek',{}],
  ]) assert.deepEqual(retrievalUsage(provider,'responses.create',request,response),expected);
  assert.deepEqual(retrievalUsage('perplexity','chat.completions.create',request,response),{retrieval_api:'sonar'});
});

test('Perplexity reads effective search tools and managed settings from extra_body',()=>{
  const response={output:[{type:'search_results'}]};
  const record=request=>providerRecord('perplexity',response,[],request);
  for (const search_type of ['web','fast']) {
    assert.equal(record({tools:[{type:'web_search'}],
      extra_body:{tools:[{type:'web_search',search_type}]}}).search_type,search_type);
  }
  assert.equal(record({extra_body:{tools:[{type:'web_search'}]}}).search_type,'web');
  for (const extra_body of [{effort:'ultra'},{effort:'low@2026-10-16.0'},{effort:'unknown'},{profile:'example-profile'}]) {
    assert.equal(record({tools:[{type:'web_search'}],extra_body}).search_type,undefined);
    assert.equal(record({tools:[{type:'web_search',search_type:'fast'}],extra_body}).search_type,'fast');
  }
});

test('Perplexity active effort levels supply fast search defaults',()=>{
  const record=request=>providerRecord('perplexity',{output:[{type:'search_results'}]},[],request);
  for (const level of ['fast','low','medium','high','xhigh']) for (const suffix of ['', '@latest']) {
    const effort=level+suffix;
    for (const settings of [{effort},{extra_body:{effort}}]) {
      assert.equal(record(settings).search_type,'fast');
      assert.equal(record({...settings,tools:[{type:'web_search'}]}).search_type,'fast');
      for (const search_type of ['web','fast']) assert.equal(record({...settings,tools:[{type:'web_search',search_type}]}).search_type,search_type);
      for (const tools of [[],[{type:'fetch_url'}]]) assert.equal(record({...settings,tools}).search_type,undefined);
    }
  }
  assert.equal(record({effort:'low',extra_body:{effort:'ultra'}}).search_type,undefined);
  assert.equal(record({extra_body:{effort:'low',tools:[{type:'web_search'}]}}).search_type,'fast');
});

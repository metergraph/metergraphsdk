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

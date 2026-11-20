const $ = id => document.getElementById(id);
let selected = null;
const el = (tag, text, cls) => { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; };
async function request(url, options){const r=await fetch(url,options); const data=await r.json(); if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail)); return data;}
function fileUrl(name){return `/api/runs/${encodeURIComponent(selected)}/files/${name.split('/').map(encodeURIComponent).join('/')}`;}
async function refresh(){
  try{
    const runs=await request('/api/runs'); $('runs').replaceChildren();
    if(!runs.length)$('runs').append(el('p','尚无任务。','muted'));
    for(const run of runs){const b=el('button',`${run.scenario_id||run.run_id} · ${run.status}`,`run ${selected===run.run_id?'active':''}`); b.append(el('small',`${run.run_id}${run.verdict?' · '+run.verdict:''}`)); b.onclick=()=>{selected=run.run_id;refresh();}; $('runs').append(b);}
    if(selected)await showDetail();
  }catch(error){$('error').textContent=error.message;}
}
async function showDetail(){
  const data=await request(`/api/runs/${selected}`); const container=$('detail'); container.replaceChildren();
  const report=data.report;
  container.append(el('p',report?`${report.verdict} · ${report.reason}`:`${data.status}${data.error?' · '+data.error:''}`,`verdict ${report?.verdict||''}`));
  $('export').hidden=['queued','running'].includes(data.status); $('export').href=`/api/runs/${selected}/export`;
  if(report){
    container.append(el('div',`操作 ${report.steps}/35 · 耗时 ${Number(report.duration_seconds).toFixed(1)} s · 模式 ${report.planner_kind||'未知'} · 策略 ${report.strategy||'未知'}`,'stats'));
    const table=el('table');const tr=el('tr');for(const title of ['检查点','状态','证据'])tr.append(el('th',title)); table.append(tr);
    for(const checkpoint of report.checkpoints||[]){const row=el('tr');row.append(el('td',checkpoint.description),el('td',checkpoint.status,checkpoint.status),el('td',(checkpoint.evidence_ids||[]).join(', ')||'尚无有效证据'));table.append(row);}container.append(table);
    container.append(el('p',`资源统计：${JSON.stringify(report.usage)}`,'hint'));
  }
  const details=el('details');details.open=!report;details.append(el('summary','最近执行事件'));details.append(el('pre',JSON.stringify(data.recent_events||[],null,2)));container.append(details);
  const images=el('div',undefined,'evidence');for(const name of (data.files||[]).filter(n=>n.endsWith('.png'))){const link=el('a',name);link.href=fileUrl(name);link.target='_blank';link.rel='noopener';const img=el('img');img.src=fileUrl(name);img.alt=name;img.loading='lazy';link.prepend(img);images.append(link);}container.append(images);
  const files=el('p');for(const name of (data.files||[]).filter(n=>!n.endsWith('.png'))){const link=el('a',name);link.href=fileUrl(name);files.append(link,document.createTextNode('  '));}container.append(files);
}
$('mode').onchange=()=>{$('scenario-id').value=$('mode').value==='demo'?'local-demo':'custom';};
$('run-form').onsubmit=async event=>{event.preventDefault();$('error').textContent='';$('submit').disabled=true;try{const data=await request('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:$('scenario-id').value,description:$('description').value,base_url:$('base-url').value,max_steps:Number($('max-steps').value),mode:$('mode').value,strategy:$('strategy').value})});selected=data.run_id;await refresh();}catch(error){$('error').textContent=error.message;}finally{$('submit').disabled=false;}};
$('refresh').onclick=refresh;
request('/api/health').then(h=>{$('health').textContent=h.model_configured?'模型接口已配置':'模型接口未配置 · 可运行演示';}).catch(()=>{$('health').textContent='服务未连接';});
refresh();setInterval(refresh,2000);

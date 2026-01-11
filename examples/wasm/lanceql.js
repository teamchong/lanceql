var en=Object.defineProperty;var tn=(a,e,t)=>e in a?en(a,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):a[e]=t;var Qe=(a,e,t)=>tn(a,typeof e!="symbol"?e+"":e,t);var V=null,Se=null,nn=0,Fe=new Map,ce="clone",Y=null,rn=16*1024*1024;function sn(){try{if(typeof SharedArrayBuffer<"u"&&typeof crossOriginIsolated<"u"&&crossOriginIsolated)return Y=new SharedArrayBuffer(rn),ce="sharedBuffer",console.log("[LanceQL] Using SharedArrayBuffer (zero-copy)"),!0}catch{}try{let a=new ArrayBuffer(8);return typeof ArrayBuffer.prototype.transfer<"u",ce="transfer",console.log("[LanceQL] Using Transferable ArrayBuffers"),!1}catch{}return ce="clone",console.log("[LanceQL] Using structured clone (fallback)"),!1}function on(){return V||(sn(),Se=new Promise((a,e)=>{console.log("[LanceQL] Using regular Worker for better logging");try{V=new Worker(new URL("./lanceql-worker.js?v="+Date.now(),import.meta.url),{type:"module",name:"lanceql"}),V.onmessage=t=>{an(t.data,V,a)},V.onerror=t=>{console.error("[LanceQL] Worker error:",t),e(t)},Y&&V.postMessage({type:"initSharedBuffer",buffer:Y})}catch(t){console.error("[LanceQL] Failed to create Worker:",t),e(t)}})),Se}function an(a,e,t){if(console.log("[LanceQL] Incoming worker message:",a.type||(a.id!==void 0?"RPC reply":"unknown")),a.type==="ready"){t(e);return}if(a.type==="log"){console.log(a.message);return}if(a.id!==void 0){let n=Fe.get(a.id);if(n)if(Fe.delete(a.id),a.sharedOffset!==void 0&&Y){let r=new Uint8Array(Y,a.sharedOffset,a.sharedLength),s=JSON.parse(new TextDecoder().decode(r));n.resolve(s)}else if(a.error)n.reject(new Error(a.error));else{let r=a.result;if(r&&r._format==="cursor"){let{cursorId:s,columns:i,rowCount:o}=r;r={_format:"columnar",columns:i,rowCount:o,_cursorId:s,_fetched:!1},Object.defineProperty(r,"data",{configurable:!0,enumerable:!0,get(){return this._fetched||console.warn("Cursor data accessed - fetching from worker"),{}}}),Object.defineProperty(r,"rows",{configurable:!0,enumerable:!0,get(){return[]}})}else if(r&&r._format==="wasm_binary"){let s=-9223372036854775808n,{buffer:i,columns:o,rowCount:c,schema:l}=r,f=new DataView(i),u=new Uint8Array(i),h=32,d=24,m={};for(let g=0;g<o.length;g++){let b=h+g*d,w=f.getUint32(b,!0),p=f.getUint32(b+8,!0),y=Number(f.getBigUint64(b+12,!0)),_=f.getUint32(b+20,!0),x=o[g];if(w<=3){let v=y/_;w===0?m[x]=new BigInt64Array(i,p,v):w===1?m[x]=new Float64Array(i,p,v):w===2?m[x]=new Int32Array(i,p,v):w===3&&(m[x]=new Float32Array(i,p,v))}else{let v=p,A=new Uint32Array(i,v,c),I=p+c*4,S=y-c*4,U=u.subarray(I,I+S),L=new TextDecoder,B=new Array(c),C=!1;m[x]=new Proxy(B,{get(k,G){if(G==="length")return c;if(typeof G=="string"&&!isNaN(G)){if(!C){for(let T=0;T<c;T++){let Ae=A[T],Ie=T<c-1?A[T+1]:S;k[T]=L.decode(U.subarray(Ae,Ie))}C=!0}return k[+G]}if(G===Symbol.iterator){if(!C){for(let T=0;T<c;T++){let Ae=A[T],Ie=T<c-1?A[T+1]:S;k[T]=L.decode(U.subarray(Ae,Ie))}C=!0}return()=>k[Symbol.iterator]()}return k[G]}})}}r={_format:"columnar",columns:o,rowCount:c,data:m},Object.defineProperty(r,"rows",{configurable:!0,enumerable:!0,get(){let g=new Array(c),b=o.map(w=>m[w]);for(let w=0;w<c;w++){let p={};for(let y=0;y<o.length;y++){let _=b[y][w];(_===s||typeof _=="number"&&isNaN(_))&&(_=null),p[o[y]]=_}g[w]=p}return Object.defineProperty(this,"rows",{value:g,writable:!1}),g}})}else if(r&&r._format==="packed"){let{columns:s,rowCount:i,packedBuffer:o,colOffsets:c,stringData:l}=r,f={...l||{}};if(o&&c){let u={Float64Array,Float32Array,Int32Array,Int16Array,Int8Array,Uint32Array,Uint16Array,Uint8Array,BigInt64Array,BigUint64Array};for(let[h,d]of Object.entries(c)){let m=u[d.type]||Float64Array;f[h]=new m(o,d.offset,d.length)}}r.data=f,r._format="columnar",Object.defineProperty(r,"rows",{configurable:!0,enumerable:!0,get(){let u=new Array(i),h=s.map(d=>f[d]);for(let d=0;d<i;d++){let m={};for(let g=0;g<s.length;g++)m[s[g]]=h[g][d];u[d]=m}return Object.defineProperty(this,"rows",{value:u,writable:!1}),u}})}else if(r&&r._format==="columnar"){let{columns:s,rowCount:i,data:o}=r;for(let c of s){let l=o[c];if(l&&l._arrowString){let{offsets:f,bytes:u,isList:h,nullable:d}=l;h&&console.log(`[WorkerRPC] Column ${c} is list mode`);let m=new TextDecoder,g=new Array(i),b=!1;o[c]=new Proxy(g,{get(w,p){if(p==="length")return i;if(typeof p=="string"&&!isNaN(p)){if(!b&&u&&f){for(let y=0;y<i;y++){let _=f[y],x=f[y+1];if(d&&_===x){w[y]=null;continue}let v=m.decode(u.subarray(_,x));try{w[y]=h?JSON.parse(v):v}catch{w[y]=v}}b=!0}return w[+p]}if(p===Symbol.iterator){if(!b&&u&&f){for(let y=0;y<i;y++){let _=f[y],x=f[y+1];if(d&&_===x){w[y]=null;continue}let v=m.decode(u.subarray(_,x));try{w[y]=h?JSON.parse(v):v}catch{w[y]=v}}b=!0}return()=>w[Symbol.iterator]()}return w[p]}})}}Object.defineProperty(r,"rows",{configurable:!0,enumerable:!0,get(){let c=new Array(i),l=s.map(f=>o[f]);for(let f=0;f<i;f++){let u={};for(let h=0;h<s.length;h++)u[s[h]]=l[h][f];c[f]=u}return Object.defineProperty(this,"rows",{value:c,writable:!1}),c}})}n.resolve(r)}}}async function F(a,e){let t=await on(),n=++nn;return new Promise((r,s)=>{Fe.set(n,{resolve:r,reject:s});let i=[];if(ce==="transfer"&&e)for(let o of Object.keys(e)){let c=e[o];c instanceof ArrayBuffer?i.push(c):ArrayBuffer.isView(c)&&i.push(c.buffer)}i.length>0?t.postMessage({id:n,method:a,args:e},i):t.postMessage({id:n,method:a,args:e})})}var Je=new Uint8Array([76,65,78,67]);var q=class a{constructor(){this.chunks=[]}static encodeVarint(e){let t=[],n=typeof e=="bigint"?e:BigInt(e);for(;n>0x7fn;)t.push(Number(n&0x7fn)|128),n>>=7n;return t.push(Number(n)),new Uint8Array(t)}static encodeFieldHeader(e,t){let n=e<<3|t;return a.encodeVarint(n)}writeVarint(e,t){this.chunks.push(a.encodeFieldHeader(e,0)),this.chunks.push(a.encodeVarint(t))}writeBytes(e,t){this.chunks.push(a.encodeFieldHeader(e,2)),this.chunks.push(a.encodeVarint(t.length)),this.chunks.push(t)}writePackedUint64(e,t){let n=[];for(let s of t)n.push(a.encodeVarint(s));let r=n.reduce((s,i)=>s+i.length,0);this.chunks.push(a.encodeFieldHeader(e,2)),this.chunks.push(a.encodeVarint(r));for(let s of n)this.chunks.push(s)}toBytes(){let e=this.chunks.reduce((r,s)=>r+s.length,0),t=new Uint8Array(e),n=0;for(let r of this.chunks)t.set(r,n),n+=r.length;return t}clear(){this.chunks=[]}},N={INT64:"int64",FLOAT64:"float64",STRING:"string",BOOL:"bool",INT32:"int32",FLOAT32:"float32"},Q=class{constructor(e={}){this.majorVersion=e.majorVersion??0,this.minorVersion=e.minorVersion??3,this.columns=[],this.rowCount=null}_validateRowCount(e){if(this.rowCount===null)this.rowCount=e;else if(this.rowCount!==e)throw new Error(`Row count mismatch: expected ${this.rowCount}, got ${e}`)}addInt64Column(e,t){this._validateRowCount(t.length),this.columns.push({name:e,type:N.INT64,data:new Uint8Array(t.buffer,t.byteOffset,t.byteLength),length:t.length})}addInt32Column(e,t){this._validateRowCount(t.length),this.columns.push({name:e,type:N.INT32,data:new Uint8Array(t.buffer,t.byteOffset,t.byteLength),length:t.length})}addFloat64Column(e,t){this._validateRowCount(t.length),this.columns.push({name:e,type:N.FLOAT64,data:new Uint8Array(t.buffer,t.byteOffset,t.byteLength),length:t.length})}addFloat32Column(e,t){this._validateRowCount(t.length),this.columns.push({name:e,type:N.FLOAT32,data:new Uint8Array(t.buffer,t.byteOffset,t.byteLength),length:t.length})}addVectorColumn(e,t,n){let r=t.length/n;if(!Number.isInteger(r))throw new Error(`Vector data length ${t.length} is not divisible by dimension ${n}`);this._validateRowCount(r),this.columns.push({name:e,type:N.FLOAT32,data:new Uint8Array(t.buffer,t.byteOffset,t.byteLength),length:r,dimension:n,isVector:!0})}addBoolColumn(e,t){this._validateRowCount(t.length);let n=new Uint8Array(t.length);for(let r=0;r<t.length;r++)n[r]=t[r]?1:0;this.columns.push({name:e,type:N.BOOL,data:n,length:t.length})}addStringColumn(e,t){this._validateRowCount(t.length);let n=new TextEncoder,r=new Int32Array(t.length+1),s=[],i=0;for(let u=0;u<t.length;u++){r[u]=i;let h=n.encode(t[u]);s.push(h),i+=h.length}r[t.length]=i;let o=new Uint8Array(r.buffer),c=s.reduce((u,h)=>u+h.length,0),l=new Uint8Array(c),f=0;for(let u of s)l.set(u,f),f+=u.length;this.columns.push({name:e,type:N.STRING,offsetsData:o,stringData:l,data:null,length:t.length})}_buildColumnMeta(e,t,n,r){let s=new q;s.writePackedUint64(1,[BigInt(e)]),s.writePackedUint64(2,[BigInt(t)]),s.writeVarint(3,n),s.writeVarint(5,0);let i=s.toBytes(),o=new q;return o.writeBytes(2,i),o.toBytes()}_buildStringColumnMeta(e,t,n,r,s){let i=new q;i.writePackedUint64(1,[BigInt(e),BigInt(n)]),i.writePackedUint64(2,[BigInt(t),BigInt(r)]),i.writeVarint(3,s),i.writeVarint(5,0);let o=i.toBytes(),c=new q;return c.writeBytes(2,o),c.toBytes()}finalize(){if(this.columns.length===0)throw new Error("No columns added");let e=[],t=0,n=[];for(let p of this.columns)if(p.type===N.STRING){let y=t;e.push(p.offsetsData),t+=p.offsetsData.length;let _=t;e.push(p.stringData),t+=p.stringData.length,n.push({type:"string",offsetsOffset:y,offsetsSize:p.offsetsData.length,dataOffset:_,dataSize:p.stringData.length,length:p.length})}else{let y=t;e.push(p.data),t+=p.data.length,n.push({type:p.type,offset:y,size:p.data.length,length:p.length})}let r=[];for(let p=0;p<this.columns.length;p++){let y=n[p],_;y.type==="string"?_=this._buildStringColumnMeta(y.offsetsOffset,y.offsetsSize,y.dataOffset,y.dataSize,y.length):_=this._buildColumnMeta(y.offset,y.size,y.length,y.type),r.push(_)}let s=t,i=[],o=0;for(let p of r)i.push(o),e.push(p),t+=p.length,o+=p.length;let c=t,l=new BigUint64Array(i.length);for(let p=0;p<i.length;p++)l[p]=BigInt(i[p]);let f=new Uint8Array(l.buffer);e.push(f),t+=f.length;let u=t,h=0,d=new ArrayBuffer(40),m=new DataView(d);m.setBigUint64(0,BigInt(s),!0),m.setBigUint64(8,BigInt(c),!0),m.setBigUint64(16,BigInt(u),!0),m.setUint32(24,h,!0),m.setUint32(28,this.columns.length,!0),m.setUint16(32,this.majorVersion,!0),m.setUint16(34,this.minorVersion,!0),new Uint8Array(d,36,4).set(Je),e.push(new Uint8Array(d));let g=t+40,b=new Uint8Array(g),w=0;for(let p of e)b.set(p,w),w+=p.length;return b}getNumColumns(){return this.columns.length}getRowCount(){return this.rowCount}getColumnNames(){return this.columns.map(e=>e.name)}};var Ze=`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LanceQL Explorer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #475569;
            --row-height: 32px;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            padding: 8px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            gap: 16px;
        }

        .logo {
            font-weight: 700;
            font-size: 14px;
            color: var(--accent);
        }

        .table-select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-primary);
            padding: 6px 12px;
            font-size: 13px;
            min-width: 200px;
        }

        .table-select:focus {
            outline: none;
            border-color: var(--accent);
        }

        .row-count {
            color: var(--text-muted);
            font-size: 12px;
            margin-left: auto;
        }

        /* Tabs */
        .tabs {
            display: flex;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
        }

        .tab {
            padding: 10px 20px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.15s;
        }

        .tab:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }

        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        /* Main content */
        .content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .tab-panel {
            display: none;
            flex: 1;
            overflow: hidden;
        }

        .tab-panel.active {
            display: flex;
            flex-direction: column;
        }

        /* SQL Tab */
        .sql-editor {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .sql-input {
            background: var(--bg-secondary);
            border: none;
            border-bottom: 1px solid var(--border);
            color: var(--text-primary);
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 13px;
            padding: 12px;
            resize: none;
            height: 120px;
        }

        .sql-input:focus {
            outline: none;
        }

        .sql-toolbar {
            display: flex;
            padding: 8px 12px;
            gap: 8px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
        }

        .btn {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 16px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.15s;
        }

        .btn:hover {
            background: var(--accent-hover);
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }

        .btn-secondary:hover {
            background: var(--border);
            color: var(--text-primary);
        }

        .sql-status {
            font-size: 12px;
            color: var(--text-muted);
            margin-left: auto;
        }

        .sql-status.error {
            color: var(--error);
        }

        .sql-status.success {
            color: var(--success);
        }

        /* Virtual Table */
        .virtual-table-container {
            flex: 1;
            overflow: hidden;
            position: relative;
        }

        .virtual-table {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            overflow: auto;
        }

        .table-header {
            display: flex;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .table-header-cell {
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            min-width: 120px;
            border-right: 1px solid var(--border);
        }

        .table-body {
            position: relative;
        }

        .table-row {
            display: flex;
            height: var(--row-height);
            border-bottom: 1px solid var(--bg-tertiary);
        }

        .table-row:hover {
            background: var(--bg-secondary);
        }

        .table-cell {
            padding: 6px 12px;
            font-size: 12px;
            font-family: 'SF Mono', Monaco, monospace;
            min-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            border-right: 1px solid var(--bg-tertiary);
            display: flex;
            align-items: center;
        }

        .table-cell.null {
            color: var(--text-muted);
            font-style: italic;
        }

        /* Timeline Tab */
        .timeline-container {
            padding: 20px;
            overflow-y: auto;
        }

        .timeline-slider {
            width: 100%;
            margin: 20px 0;
        }

        .timeline-versions {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .timeline-version {
            display: flex;
            align-items: center;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all 0.15s;
        }

        .timeline-version:hover {
            border-color: var(--accent);
        }

        .timeline-version.active {
            border-color: var(--accent);
            background: rgba(59, 130, 246, 0.1);
        }

        .version-number {
            font-weight: 700;
            font-size: 14px;
            width: 60px;
        }

        .version-meta {
            flex: 1;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .version-delta {
            font-weight: 600;
            font-size: 13px;
        }

        .version-delta.add {
            color: var(--success);
        }

        .version-delta.delete {
            color: var(--error);
        }

        /* Diff View */
        .diff-view {
            display: flex;
            gap: 20px;
            padding: 20px;
            overflow-y: auto;
        }

        .diff-panel {
            flex: 1;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }

        .diff-header {
            padding: 12px;
            font-weight: 600;
            font-size: 13px;
            border-bottom: 1px solid var(--border);
        }

        .diff-header.added {
            background: rgba(34, 197, 94, 0.1);
            color: var(--success);
        }

        .diff-header.deleted {
            background: rgba(239, 68, 68, 0.1);
            color: var(--error);
        }

        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
        }

        .spinner {
            width: 24px;
            height: 24px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 12px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Empty state */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <span class="logo">LanceQL Explorer</span>
        <select class="table-select" id="tableSelect">
            <option value="">Select a table...</option>
        </select>
        <span class="row-count" id="rowCount"></span>
    </div>

    <div class="tabs">
        <div class="tab active" data-tab="sql">SQL</div>
        <div class="tab" data-tab="dataview">DataView</div>
        <div class="tab" data-tab="timeline">Timeline</div>
    </div>

    <div class="content">
        <!-- SQL Tab -->
        <div class="tab-panel active" id="tab-sql">
            <div class="sql-editor">
                <textarea class="sql-input" id="sqlInput" placeholder="SELECT * FROM table LIMIT 100"></textarea>
                <div class="sql-toolbar">
                    <button class="btn" id="runSql">Run (Ctrl+Enter)</button>
                    <button class="btn btn-secondary" id="clearSql">Clear</button>
                    <span class="sql-status" id="sqlStatus"></span>
                </div>
                <div class="virtual-table-container" id="sqlResults">
                    <div class="empty-state">Run a query to see results</div>
                </div>
            </div>
        </div>

        <!-- DataView Tab -->
        <div class="tab-panel" id="tab-dataview">
            <div class="virtual-table-container" id="dataviewContainer">
                <div class="empty-state">Select a table to view data</div>
            </div>
        </div>

        <!-- Timeline Tab -->
        <div class="tab-panel" id="tab-timeline">
            <div class="timeline-container" id="timelineContainer">
                <div class="empty-state">Select a table to view version history</div>
            </div>
        </div>
    </div>

    <script>
        // =====================================================================
        // Explorer Logic
        // =====================================================================

        const state = {
            tables: [],
            currentTable: null,
            currentVersion: null,
            totalRows: 0,
            columns: [],
            cache: new Map(), // row cache
            pendingRequests: new Map(),
            requestId: 0
        };

        // Elements
        const tableSelect = document.getElementById('tableSelect');
        const rowCount = document.getElementById('rowCount');
        const sqlInput = document.getElementById('sqlInput');
        const sqlStatus = document.getElementById('sqlStatus');
        const sqlResults = document.getElementById('sqlResults');
        const dataviewContainer = document.getElementById('dataviewContainer');
        const timelineContainer = document.getElementById('timelineContainer');

        // =====================================================================
        // Communication with parent window
        // =====================================================================

        function sendMessage(type, params = {}) {
            return new Promise((resolve, reject) => {
                const id = ++state.requestId;
                state.pendingRequests.set(id, { resolve, reject });
                window.opener.postMessage({ type, id, ...params }, '*');

                // Timeout after 30s
                setTimeout(() => {
                    if (state.pendingRequests.has(id)) {
                        state.pendingRequests.delete(id);
                        reject(new Error('Request timeout'));
                    }
                }, 30000);
            });
        }

        window.addEventListener('message', (event) => {
            const { type, id, result, error, ...data } = event.data || {};

            if (type === 'response') {
                const pending = state.pendingRequests.get(id);
                if (pending) {
                    state.pendingRequests.delete(id);
                    if (error) {
                        pending.reject(new Error(error));
                    } else {
                        pending.resolve(result);
                    }
                }
            } else if (type === 'init') {
                initExplorer(data);
            } else if (type === 'run-sql') {
                sqlInput.value = data.sql;
                runQuery();
            } else if (type === 'select-table') {
                tableSelect.value = data.table;
                loadTable(data.table);
            }
        });

        // Signal ready
        window.opener?.postMessage({ type: 'ready' }, '*');

        // =====================================================================
        // Initialization
        // =====================================================================

        function initExplorer(data) {
            state.tables = data.tables || [];

            // Populate table select
            tableSelect.innerHTML = '<option value="">Select a table...</option>';
            for (const table of state.tables) {
                const opt = document.createElement('option');
                opt.value = table;
                opt.textContent = table;
                tableSelect.appendChild(opt);
            }

            // Auto-select table if provided
            if (data.options?.table) {
                tableSelect.value = data.options.table;
                loadTable(data.options.table);
            }
        }

        // =====================================================================
        // Tab switching
        // =====================================================================

        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

                tab.classList.add('active');
                document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
            });
        });

        // =====================================================================
        // Table selection
        // =====================================================================

        tableSelect.addEventListener('change', () => {
            const table = tableSelect.value;
            if (table) loadTable(table);
        });

        async function loadTable(table) {
            state.currentTable = table;
            state.cache.clear();

            try {
                // Get schema and count
                const [schema, count] = await Promise.all([
                    sendMessage('fetch-schema', { table }),
                    sendMessage('fetch-count', { table })
                ]);

                state.columns = schema.columns;
                state.totalRows = count;
                rowCount.textContent = count.toLocaleString() + ' rows';

                // Render DataView
                renderVirtualTable(dataviewContainer, table);

                // Load timeline
                loadTimeline(table);

            } catch (e) {
                console.error('Failed to load table:', e);
                rowCount.textContent = 'Error loading table';
            }
        }

        // =====================================================================
        // Virtual Table Rendering
        // =====================================================================

        function renderVirtualTable(container, table, data = null) {
            const ROW_HEIGHT = 32;
            const BUFFER_ROWS = 5;
            const MIN_VISIBLE = 20;
            const MAX_VISIBLE = 200;

            // Calculate visible rows based on container height
            const containerHeight = container.clientHeight || 400;
            const calculatedRows = Math.ceil(containerHeight / ROW_HEIGHT);
            const VISIBLE_ROWS = Math.min(MAX_VISIBLE, Math.max(MIN_VISIBLE, calculatedRows + BUFFER_ROWS * 2));

            if (!state.columns.length && !data) {
                container.innerHTML = '<div class="empty-state">No columns</div>';
                return;
            }

            const columns = data?.columns || state.columns;
            const totalRows = data?.rows?.length || state.totalRows;

            container.innerHTML = \`
                <div class="virtual-table" id="virtualTable">
                    <div class="table-header">
                        \${columns.map(col => \`<div class="table-header-cell">\${col}</div>\`).join('')}
                    </div>
                    <div class="table-body" style="height: \${totalRows * ROW_HEIGHT}px">
                    </div>
                </div>
            \`;

            const virtualTable = container.querySelector('#virtualTable');
            const tableBody = container.querySelector('.table-body');

            let lastScrollTop = 0;
            let renderedStart = -1;
            let renderedEnd = -1;

            async function renderRows() {
                const scrollTop = virtualTable.scrollTop;
                const startRow = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER_ROWS);
                const endRow = Math.min(totalRows, startRow + VISIBLE_ROWS);

                // Skip if same range
                if (startRow === renderedStart && endRow === renderedEnd) return;

                renderedStart = startRow;
                renderedEnd = endRow;

                // Fetch rows if needed
                let rows;
                if (data) {
                    rows = data.rows.slice(startRow, endRow);
                } else {
                    const cacheKey = \`\${startRow}-\${endRow}\`;
                    if (state.cache.has(cacheKey)) {
                        rows = state.cache.get(cacheKey);
                    } else {
                        try {
                            const result = await sendMessage('fetch-rows', {
                                table,
                                offset: startRow,
                                limit: endRow - startRow
                            });
                            rows = result?.rows || [];
                            state.cache.set(cacheKey, rows);
                        } catch (e) {
                            console.error('Failed to fetch rows:', e);
                            rows = [];
                        }
                    }
                }

                // Render rows
                const rowsHtml = rows.map((row, i) => {
                    const rowIndex = startRow + i;
                    const cells = columns.map((col, j) => {
                        const value = row[j];
                        const isNull = value === null || value === undefined;
                        return \`<div class="table-cell \${isNull ? 'null' : ''}">\${isNull ? 'NULL' : escapeHtml(String(value))}</div>\`;
                    }).join('');
                    return \`<div class="table-row" style="position: absolute; top: \${rowIndex * ROW_HEIGHT}px; width: 100%">\${cells}</div>\`;
                }).join('');

                tableBody.innerHTML = rowsHtml;
            }

            virtualTable.addEventListener('scroll', () => {
                requestAnimationFrame(renderRows);
            });

            // Re-render on resize
            const resizeObserver = new ResizeObserver(() => {
                requestAnimationFrame(renderRows);
            });
            resizeObserver.observe(container);

            // Initial render
            renderRows();
        }

        // =====================================================================
        // SQL Execution
        // =====================================================================

        async function runQuery() {
            const sql = sqlInput.value.trim();
            if (!sql) return;

            sqlStatus.textContent = 'Running...';
            sqlStatus.className = 'sql-status';

            const startTime = performance.now();

            try {
                const result = await sendMessage('exec', { sql });
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

                if (result?.columns && result?.rows) {
                    sqlStatus.textContent = \`\${result.rows.length} rows in \${elapsed}s\`;
                    sqlStatus.className = 'sql-status success';

                    // Render results
                    state.columns = result.columns;
                    renderVirtualTable(sqlResults, null, result);
                } else {
                    sqlStatus.textContent = \`Done in \${elapsed}s\`;
                    sqlStatus.className = 'sql-status success';
                    sqlResults.innerHTML = '<div class="empty-state">Query executed successfully</div>';
                }
            } catch (e) {
                sqlStatus.textContent = e.message;
                sqlStatus.className = 'sql-status error';
            }
        }

        document.getElementById('runSql').addEventListener('click', runQuery);
        document.getElementById('clearSql').addEventListener('click', () => {
            sqlInput.value = '';
            sqlResults.innerHTML = '<div class="empty-state">Run a query to see results</div>';
            sqlStatus.textContent = '';
        });

        sqlInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                runQuery();
            }
        });

        // =====================================================================
        // Timeline
        // =====================================================================

        async function loadTimeline(table) {
            try {
                const versions = await sendMessage('timeline', { table });
                renderTimeline(versions);
            } catch (e) {
                timelineContainer.innerHTML = '<div class="empty-state">Failed to load timeline</div>';
            }
        }

        function renderTimeline(versions) {
            if (!versions.length) {
                timelineContainer.innerHTML = '<div class="empty-state">No version history</div>';
                return;
            }

            timelineContainer.innerHTML = \`
                <h3 style="margin-bottom: 16px; font-size: 14px; color: var(--text-secondary)">Version History</h3>
                <div class="timeline-versions">
                    \${versions.map(v => \`
                        <div class="timeline-version" data-version="\${v.version}">
                            <span class="version-number">v\${v.version}</span>
                            <span class="version-meta">\${v.operation} - \${new Date(v.timestamp).toLocaleString()}</span>
                            <span class="version-delta \${v.delta.startsWith('+') ? 'add' : 'delete'}">\${v.delta}</span>
                        </div>
                    \`).join('')}
                </div>
            \`;

            // Click to view version
            timelineContainer.querySelectorAll('.timeline-version').forEach(el => {
                el.addEventListener('click', () => {
                    const version = el.dataset.version;
                    document.querySelectorAll('.timeline-version').forEach(v => v.classList.remove('active'));
                    el.classList.add('active');

                    // Switch to dataview with this version
                    state.currentVersion = parseInt(version);
                    document.querySelector('[data-tab="dataview"]').click();
                    // Re-render with version
                    if (state.currentTable) {
                        sqlInput.value = \`SELECT * FROM \${state.currentTable} VERSION AS OF \${version} LIMIT 100\`;
                    }
                });
            });
        }

        // =====================================================================
        // Utils
        // =====================================================================

        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
    <\/script>
</body>
</html>`;var Ue=class{constructor(e,t,n){this._vault=e,this._window=t,this._options=n,this._messageHandler=null,this._setupMessageHandler()}_setupMessageHandler(){this._messageHandler=async e=>{if(e.source!==this._window)return;let{type:t,id:n,...r}=e.data||{};if(t)try{let s;switch(t){case"exec":s=await this._vault.exec(r.sql);break;case"query":s=await this._vault.query(r.sql);break;case"tables":s=await this._vault.tables();break;case"fetch-rows":s=await this._fetchRows(r);break;case"fetch-schema":s=await this._fetchSchema(r.table);break;case"fetch-count":s=await this._fetchCount(r.table);break;case"timeline":s=await this._fetchTimeline(r.table);break;case"diff":s=await this._fetchDiff(r);break;case"ready":this._sendInitialData();return;default:console.warn("Unknown explorer message type:",t);return}this._window.postMessage({type:"response",id:n,result:s},"*")}catch(s){this._window.postMessage({type:"response",id:n,error:s.message},"*")}},window.addEventListener("message",this._messageHandler)}async _fetchRows({table:e,offset:t,limit:n,version:r}){let s=`SELECT * FROM ${e}`;return r&&(s+=` VERSION AS OF ${r}`),s+=` LIMIT ${n} OFFSET ${t}`,await this._vault.exec(s)}async _fetchSchema(e){return{columns:(await this._vault.exec(`SELECT * FROM ${e} LIMIT 0`))?.columns||[]}}async _fetchCount(e){return(await this._vault.exec(`SELECT COUNT(*) FROM ${e}`))?.rows?.[0]?.[0]||0}async _fetchTimeline(e){try{return(await this._vault.exec(`SHOW VERSIONS FOR ${e}`))?.rows||[]}catch{return[{version:1,timestamp:Date.now(),operation:"CURRENT",rowCount:0,delta:"+0"}]}}async _fetchDiff({table:e,from:t,to:n}){try{return await this._vault.exec(`DIFF ${e} VERSION ${t} AND VERSION ${n}`)}catch(r){return{added:[],deleted:[],error:r.message}}}async _sendInitialData(){let e=await this._vault.tables();this._window.postMessage({type:"init",tables:e,options:this._options},"*")}runSQL(e){this._window.postMessage({type:"run-sql",sql:e},"*")}selectTable(e){this._window.postMessage({type:"select-table",table:e},"*")}close(){window.removeEventListener("message",this._messageHandler),this._window&&!this._window.closed&&this._window.close()}};function et(a,e={}){let{width:t=1200,height:n=800,table:r=null,version:s=null}=e,i=(screen.width-t)/2,o=(screen.height-n)/2,c=window.open("","lanceql-explorer",`width=${t},height=${n},left=${i},top=${o},menubar=no,toolbar=no,location=no,status=no`);if(!c)throw new Error("Failed to open explorer window. Check popup blocker settings.");return c.document.write(Ze),c.document.close(),new Ue(a,c,{table:r,version:s})}var le=class{constructor(e=null){this._getEncryptionKey=e,this._encryptionKeyId=null,this._ready=!1}async _init(){if(this._ready)return this;let e=null;if(this._getEncryptionKey){let t=await this._getEncryptionKey();this._encryptionKeyId=`vault:${Date.now()}`;let n;if(t instanceof CryptoKey)n=await crypto.subtle.exportKey("raw",t);else if(t instanceof ArrayBuffer||t instanceof Uint8Array)n=t instanceof Uint8Array?t:new Uint8Array(t);else if(typeof t=="string"){let s=new TextEncoder().encode(t),i=await crypto.subtle.digest("SHA-256",s);n=new Uint8Array(i)}else throw new Error("Encryption key must be CryptoKey, ArrayBuffer, Uint8Array, or string");e={keyId:this._encryptionKeyId,keyBytes:Array.from(n instanceof Uint8Array?n:new Uint8Array(n))}}return await F("vault:open",{encryption:e}),this._ready=!0,this}async get(e){return F("vault:get",{key:e})}async set(e,t){await F("vault:set",{key:e,value:t})}async delete(e){return F("vault:delete",{key:e})}async keys(){return F("vault:keys",{})}async has(e){return await this.get(e)!==void 0}async exec(e){return F("vault:exec",{sql:e})}async query(e){let t=await this.exec(e);return!t||!t.columns||!t.rows?[]:t.rows.map(n=>{let r={};return t.columns.forEach((s,i)=>{r[s]=n[i]}),r})}table(e){return new ue(this,e)}async tables(){return F("vault:tables",{})}async df(e){let t=await this.exec(e);if(!t||!t.columns||!t.rows)return new J({},[]);let n={},r=t.columns;for(let s of r)n[s]=[];for(let s of t.rows)for(let i=0;i<r.length;i++){let o=r[i];n[o].push(s[o]!==void 0?s[o]:s[i])}return new J(n,r)}async exportToLance(e){let t=await this.exec(`SELECT * FROM ${e} LIMIT 0`);if(!t||!t.columns)throw new Error(`Table '${e}' not found or empty`);let n=await this.exec(`SELECT * FROM ${e}`);if(!n||!n.rows||n.rows.length===0)throw new Error(`Table '${e}' is empty`);let r=new Q,s=n.columns,i=n.rows;for(let o=0;o<s.length;o++){let c=s[o],l=i.map(u=>u[c]!==void 0?u[c]:u[o]),f=l.find(u=>u!=null);if(f===void 0)r.addStringColumn(c,l.map(u=>u===null?"":String(u)));else if(typeof f=="bigint")r.addInt64Column(c,BigInt64Array.from(l.map(u=>u===null?0n:BigInt(u))));else if(typeof f=="number")Number.isInteger(f)&&f<=2147483647&&f>=-2147483648?r.addInt32Column(c,Int32Array.from(l.map(u=>u===null?0:u))):r.addFloat64Column(c,Float64Array.from(l.map(u=>u===null?0:u)));else if(typeof f=="boolean")r.addBoolColumn(c,l.map(u=>u===null?!1:u));else if(Array.isArray(f)){let u=f.length,h=new Float32Array(l.length*u);for(let d=0;d<l.length;d++){let m=l[d]||new Array(u).fill(0);for(let g=0;g<u;g++)h[d*u+g]=m[g]||0}r.addVectorColumn(c,h,u)}else r.addStringColumn(c,l.map(u=>u===null?"":String(u)))}return r.finalize()}async uploadToUrl(e,t,n={}){let{onProgress:r}=n;if(r&&typeof XMLHttpRequest<"u")return new Promise((i,o)=>{let c=new XMLHttpRequest;c.open("PUT",t,!0),c.setRequestHeader("Content-Type","application/octet-stream"),c.upload.onprogress=l=>{l.lengthComputable&&r(l.loaded,l.total)},c.onload=()=>{c.status>=200&&c.status<300?i({ok:!0,status:c.status}):o(new Error(`Upload failed: ${c.status} ${c.statusText}`))},c.onerror=()=>o(new Error("Upload failed: network error")),c.send(e)});let s=await fetch(t,{method:"PUT",body:e,headers:{"Content-Type":"application/octet-stream"}});if(!s.ok)throw new Error(`Upload failed: ${s.status} ${s.statusText}`);return s}async exportToRemote(e,t,n={}){let r=await this.exportToLance(e);return await this.uploadToUrl(r,t,n),{size:r.length,url:t.split("?")[0]}}explorer(e={}){return et(this,e)}async timeline(e){return(await this.exec(`SHOW VERSIONS FOR ${e}`))?.rows?.map(n=>({version:n[0],timestamp:n[1],operation:n[2],rowCount:n[3],delta:n[4]}))||[]}async diff(e,{from:t,to:n,limit:r=100}={}){let s=`DIFF ${e} VERSION ${t}`;n!==void 0&&(s+=` AND VERSION ${n}`),s+=` LIMIT ${r}`;let i=await this.exec(s);return{added:i?.added||[],deleted:i?.deleted||[],fromVersion:i?.from_version||t,toVersion:i?.to_version||n||"HEAD"}}async lastChange(e){return this.diff(e,{from:-1})}},ue=class a{constructor(e,t){this._vault=e,this._tableName=t,this._filters=[],this._similar=null,this._selectCols=null,this._limitValue=null,this._orderBy=null}filter(e,t,n){let r=this._clone();return r._filters.push({column:e,op:t,value:n}),r}similar(e,t,n=20){let r=this._clone();return r._similar={column:e,text:t,limit:n},r}select(...e){let t=this._clone();return t._selectCols=e.flat(),t}limit(e){let t=this._clone();return t._limitValue=e,t}orderBy(e,t="ASC"){let n=this._clone();return n._orderBy={column:e,direction:t},n}async toArray(){let e=this._toSQL();return this._vault.query(e)}async first(){let e=this._clone();return e._limitValue=1,(await e.toArray())[0]||null}async count(){let e=this._toSQL(!0);return(await this._vault.exec(e))?.rows?.[0]?.[0]||0}_toSQL(e=!1){let n=`SELECT ${e?"COUNT(*)":this._selectCols?.join(", ")||"*"} FROM ${this._tableName}`,r=[];for(let i of this._filters){let o=typeof i.value=="string"?`'${i.value}'`:i.value;r.push(`${i.column} ${i.op} ${o}`)}this._similar&&r.push(`${this._similar.column} NEAR '${this._similar.text}'`),r.length>0&&(n+=" WHERE "+r.join(" AND ")),this._orderBy&&!e&&(n+=` ORDER BY ${this._orderBy.column} ${this._orderBy.direction}`);let s=this._similar?.limit||this._limitValue;return s&&!e&&(n+=` LIMIT ${s}`),n}_clone(){let e=new a(this._vault,this._tableName);return e._filters=[...this._filters],e._similar=this._similar,e._selectCols=this._selectCols?[...this._selectCols]:null,e._limitValue=this._limitValue,e._orderBy=this._orderBy,e}},J=class a{constructor(e,t){this._columns=e,this._columnNames=t,this._rowCount=t.length>0&&e[t[0]]?.length||0}get rowCount(){return this._rowCount}get columns(){return[...this._columnNames]}static fromRecords(e){if(!e||e.length===0)return new a({},[]);let t=Object.keys(e[0]),n={};for(let r of t)n[r]=[];for(let r of e)for(let s of t)n[s].push(r[s]!==void 0?r[s]:null);return new a(n,t)}static fromColumns(e){let t=Object.keys(e),n={};for(let r of t){let s=e[r];ArrayBuffer.isView(s)&&!(s instanceof Uint8Array)?n[r]=Array.from(s):n[r]=Array.isArray(s)?s:[s]}return new a(n,t)}async saveTo(e,t,n={}){let{ifExists:r="fail"}=n;switch(r){case"replace":return await e.exec(`DROP TABLE IF EXISTS ${t}`),await this._createTableAndInsert(e,t),{rowCount:this._rowCount,tableName:t};case"append":try{return await this._insertInto(e,t),{rowCount:this._rowCount,tableName:t}}catch(s){if(s.message&&s.message.includes("not found"))return await this._createTableAndInsert(e,t),{rowCount:this._rowCount,tableName:t};throw s}default:try{return await this._createTableAndInsert(e,t),{rowCount:this._rowCount,tableName:t}}catch(s){throw s.message&&(s.message.includes("AlreadyExists")||s.message.includes("already exists"))?new Error(`Table '${t}' already exists`):s}}}_inferType(e){for(let t of e)if(t!=null)return typeof t=="bigint"?"INT64":typeof t=="number"?Number.isInteger(t)&&t<=2147483647&&t>=-2147483648?"INT":"FLOAT64":typeof t=="boolean"?"INT":Array.isArray(t)?`FLOAT32[${t.length}]`:"TEXT";return"TEXT"}_formatValue(e){return e==null?"NULL":typeof e=="number"||typeof e=="bigint"?String(e):typeof e=="boolean"?e?"1":"0":Array.isArray(e)?`[${e.join(",")}]`:`'${String(e).replace(/'/g,"''")}'`}async _createTableAndInsert(e,t){let n=this._columnNames.map(r=>{let s=this._inferType(this._columns[r]);return`${r} ${s}`}).join(", ");await e.exec(`CREATE TABLE ${t} (${n})`),await this._insertInto(e,t)}async _insertInto(e,t){if(this._rowCount===0)return{rowCount:0};let n=this._columnNames.join(", "),r=[];for(let s=0;s<this._rowCount;s++){let i=this._columnNames.map(o=>this._formatValue(this._columns[o][s])).join(", ");r.push(`(${i})`)}return await e.exec(`INSERT INTO ${t} (${n}) VALUES ${r.join(", ")}`),{rowCount:this._rowCount}}row(e){if(e<0||e>=this._rowCount)return null;let t={};for(let n of this._columnNames)t[n]=this._columns[n][e];return t}column(e){return this._columns[e]||null}toRecords(){let e=[];for(let t=0;t<this._rowCount;t++)e.push(this.row(t));return e}};async function tt(a=null){let e=new le(a);return await e._init(),e}function fe(a){let e=0;for(let t=0;t<a.length;t++)e=(e<<5)-e+a.charCodeAt(t),e|=0;return e.toString(36)}var he=class{constructor(e,t=256*1024*1024){this.device=e,this.maxPoolSize=t,this.currentSize=0,this.buffers=new Map,this.accessOrder=[]}getOrCreate(e,t,n=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST){let r=fe(e),s=this.buffers.get(r);if(s){if(this._touch(r),s.size===t.byteLength)return s.buffer;this.invalidate(e)}this._ensureSpace(t.byteLength);let i=this.device.createBuffer({size:t.byteLength,usage:n,mappedAtCreation:!0}),o=t.constructor;return new o(i.getMappedRange()).set(t),i.unmap(),this.buffers.set(r,{buffer:i,size:t.byteLength,lastAccess:Date.now(),usage:n,originalKey:e}),this.currentSize+=t.byteLength,this.accessOrder.push(r),i}getStorageBuffer(e,t){return this.getOrCreate(e,t,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST)}getUniformBuffer(e,t){return this.getOrCreate(e,t,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST)}has(e){return this.buffers.has(fe(e))}get(e){let t=fe(e),n=this.buffers.get(t);return n?(this._touch(t),n.buffer):null}invalidate(e){let t=fe(e),n=this.buffers.get(t);n&&(n.buffer.destroy(),this.currentSize-=n.size,this.buffers.delete(t),this.accessOrder=this.accessOrder.filter(r=>r!==t))}invalidatePrefix(e){let t=[];for(let[n,r]of this.buffers)r.originalKey.startsWith(e)&&t.push(n);for(let n of t){let r=this.buffers.get(n);r.buffer.destroy(),this.currentSize-=r.size,this.buffers.delete(n)}this.accessOrder=this.accessOrder.filter(n=>!t.includes(n))}clear(){for(let e of this.buffers.values())e.buffer.destroy();this.buffers.clear(),this.accessOrder=[],this.currentSize=0}stats(){return{entries:this.buffers.size,currentSize:this.currentSize,maxSize:this.maxPoolSize,utilization:this.currentSize/this.maxPoolSize}}_touch(e){let t=this.accessOrder.indexOf(e);t!==-1&&this.accessOrder.splice(t,1),this.accessOrder.push(e);let n=this.buffers.get(e);n&&(n.lastAccess=Date.now())}_ensureSpace(e){for(;this.currentSize+e>this.maxPoolSize&&this.accessOrder.length>0;){let t=this.accessOrder.shift(),n=this.buffers.get(t);n&&(n.buffer.destroy(),this.currentSize-=n.size,this.buffers.delete(t))}}},nt=new WeakMap;function M(a,e=256*1024*1024){let t=nt.get(a);return t||(t=new he(a,e),nt.set(a,t)),t}var de=class{constructor(){this.device=null,this.pipeline=null,this.available=!1,this._initPromise=null}async init(){return this._initPromise?this._initPromise:(this._initPromise=this._doInit(),this._initPromise)}async _doInit(){if(!navigator.gpu)return console.log("[WebGPU] Not available in this browser"),!1;try{let e=await navigator.gpu.requestAdapter();return e?(this.device=await e.requestDevice(),this._createPipeline(),this.available=!0,console.log("[WebGPU] Initialized successfully"),!0):(console.log("[WebGPU] No adapter found"),!1)}catch(e){return console.warn("[WebGPU] Init failed:",e),!1}}_createPipeline(){let t=this.device.createShaderModule({code:`
            struct Params {
                dim: u32,
                numVectors: u32,
            }

            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read> query: array<f32>;
            @group(0) @binding(2) var<storage, read> vectors: array<f32>;
            @group(0) @binding(3) var<storage, read_write> scores: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) globalId: vec3u) {
                let idx = globalId.x;
                if (idx >= params.numVectors) {
                    return;
                }

                let dim = params.dim;
                let offset = idx * dim;

                // Compute dot product (= cosine similarity for normalized vectors)
                var dot: f32 = 0.0;
                for (var i: u32 = 0u; i < dim; i++) {
                    dot += query[i] * vectors[offset + i];
                }

                scores[idx] = dot;
            }
        `});this.pipeline=this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"main"}})}async batchCosineSimilarity(e,t,n=!0,r=!1){if(!this.available||t.length===0)return null;let s=e.length,i=r?t.length/s:t.length,o=i*s*4,c=this.device.limits?.maxStorageBufferBindingSize||134217728;if(o>c)return console.warn(`[WebGPU] Buffer size ${(o/1024/1024).toFixed(1)}MB exceeds limit ${(c/1024/1024).toFixed(1)}MB, falling back`),null;let l=this.device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),f=this.device.createBuffer({size:s*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),u=this.device.createBuffer({size:i*s*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),h=this.device.createBuffer({size:i*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),d=this.device.createBuffer({size:i*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});if(this.device.queue.writeBuffer(l,0,new Uint32Array([s,i])),this.device.queue.writeBuffer(f,0,e),r)this.device.queue.writeBuffer(u,0,t);else{let p=new Float32Array(i*s);for(let y=0;y<i;y++)p.set(t[y],y*s);this.device.queue.writeBuffer(u,0,p)}let m=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:h}}]}),g=this.device.createCommandEncoder(),b=g.beginComputePass();b.setPipeline(this.pipeline),b.setBindGroup(0,m),b.dispatchWorkgroups(Math.ceil(i/256)),b.end(),g.copyBufferToBuffer(h,0,d,0,i*4),this.device.queue.submit([g.finish()]),await d.mapAsync(GPUMapMode.READ);let w=new Float32Array(d.getMappedRange().slice(0));return d.unmap(),l.destroy(),f.destroy(),u.destroy(),h.destroy(),d.destroy(),w}isAvailable(){return this.available}getMaxVectorsPerBatch(e){if(!this.available)return 0;let t=this.device.limits?.maxStorageBufferBindingSize||134217728;return Math.floor(t*.9/(e*4))}},Pe=null;function O(){return Pe||(Pe=new de),Pe}var H={F32:0,F16:1,Q4_0:2,Q4_1:3,Q5_0:6,Q5_1:7,Q8_0:8,Q8_1:9,Q2_K:10,Q3_K:11,Q4_K:12,Q5_K:13,Q6_K:14,Q8_K:15,I8:16,I16:17,I32:18,I64:19,F64:20,BF16:21},gr={[H.F32]:4,[H.F16]:2,[H.BF16]:2,[H.Q8_0]:33,[H.Q4_0]:18,[H.Q4_1]:20};var hn=`
struct ReduceParams {
    size: u32,
    workgroups: u32,
}

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(0.0, input[gid.x], gid.x < params.size);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] += shared_data[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { output[wid.x] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_sum_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(0.0, input[tid], tid < params.workgroups);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] += shared_data[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { output[0] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_min(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(3.4e+38, input[gid.x], gid.x < params.size);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = min(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[wid.x] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_min_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(3.4e+38, input[tid], tid < params.workgroups);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = min(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[0] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(-3.4e+38, input[gid.x], gid.x < params.size);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = max(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[wid.x] = shared_data[0]; }
}

@compute @workgroup_size(256)
fn reduce_max_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    shared_data[tid] = select(-3.4e+38, input[tid], tid < params.workgroups);
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { shared_data[tid] = max(shared_data[tid], shared_data[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { output[0] = shared_data[0]; }
}
`,Te=1e4,me=class{constructor(){this.device=null,this.pipelines=new Map,this.available=!1,this.bufferPool=null}async init(){if(this.device)return this.available;if(typeof navigator>"u"||!navigator.gpu)return console.log("[GPUAggregator] WebGPU not available"),!1;try{let e=await navigator.gpu.requestAdapter();return e?(this.device=await e.requestDevice({requiredLimits:{maxStorageBufferBindingSize:256*1024*1024,maxBufferSize:256*1024*1024}}),await this._compileShaders(),this.bufferPool=M(this.device),this.available=!0,console.log("[GPUAggregator] Initialized"),!0):(console.log("[GPUAggregator] No WebGPU adapter"),!1)}catch(e){return console.error("[GPUAggregator] Init failed:",e),!1}}isAvailable(){return this.available}async _compileShaders(){let e=this.device.createShaderModule({code:hn});for(let t of["sum","min","max"])this.pipelines.set(`reduce_${t}`,this.device.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:`reduce_${t}`}})),this.pipelines.set(`reduce_${t}_final`,this.device.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:`reduce_${t}_final`}}))}async sum(e,t){return e.length<Te||!this.available?this._cpuSum(e):this._gpuReduce(e,"sum",t)}count(e){return e.length}async avg(e){return e.length===0?null:await this.sum(e)/e.length}async min(e,t){return e.length===0?null:e.length<Te||!this.available?this._cpuMin(e):this._gpuReduce(e,"min",t)}async max(e,t){return e.length===0?null:e.length<Te||!this.available?this._cpuMax(e):this._gpuReduce(e,"max",t)}async batch(e,t){let n={};for(let r of t)switch(r){case"sum":n.sum=await this.sum(e);break;case"count":n.count=await this.count(e);break;case"avg":n.sum!==void 0?n.avg=e.length>0?n.sum/e.length:null:n.avg=await this.avg(e);break;case"min":n.min=await this.min(e);break;case"max":n.max=await this.max(e);break}return n}async _gpuReduce(e,t,n){let r=e.length,i=Math.ceil(r/256),o=e instanceof Float32Array?e:new Float32Array(e),c,l=!1;n&&this.bufferPool?(c=this.bufferPool.getStorageBuffer(n,o),l=!0):(c=this.device.createBuffer({size:o.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(c,0,o));let f=this.device.createBuffer({size:i*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),u=this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),h=this.device.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),d=new Uint32Array([r,i]),m=this.device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(m,0,d);let g=this.pipelines.get(`reduce_${t}`),b=this.device.createBindGroup({layout:g.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:f}}]}),w=new Uint32Array([i,i]),p=this.device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(p,0,w);let y=this.pipelines.get(`reduce_${t}_final`),_=this.device.createBindGroup({layout:y.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:u}}]}),x=this.device.createCommandEncoder(),v=x.beginComputePass();v.setPipeline(g),v.setBindGroup(0,b),v.dispatchWorkgroups(i),v.end();let A=x.beginComputePass();A.setPipeline(y),A.setBindGroup(0,_),A.dispatchWorkgroups(1),A.end(),x.copyBufferToBuffer(u,0,h,0,4),this.device.queue.submit([x.finish()]),await h.mapAsync(GPUMapMode.READ);let I=new Float32Array(h.getMappedRange())[0];return h.unmap(),l||c.destroy(),f.destroy(),u.destroy(),h.destroy(),m.destroy(),p.destroy(),I}_cpuSum(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}_cpuMin(e){let t=e[0];for(let n=1;n<e.length;n++)e[n]<t&&(t=e[n]);return t}_cpuMax(e){let t=e[0];for(let n=1;n<e.length;n++)e[n]>t&&(t=e[n]);return t}getBufferPool(){return this.bufferPool}invalidateTable(e){this.bufferPool&&this.bufferPool.invalidatePrefix(e+":")}dispose(){this.bufferPool&&(this.bufferPool.clear(),this.bufferPool=null),this.pipelines.clear(),this.device=null,this.available=!1}},Ce=null;function rt(){return Ce||(Ce=new me),Ce}var st=".lanceql-buffers";var ge=class{constructor(e,t={}){this.name=e,this.maxMemory=t.maxMemory||16*1024*1024,this._file=null,this._syncHandle=null,this._writeOffset=0,this._memBuffer=[],this._memSize=0,this._finalized=!1,this._totalEntries=0,this._entrySize=4}async init(e=4){this._entrySize=e;try{let n=await(await navigator.storage.getDirectory()).getDirectoryHandle(st,{create:!0});try{await n.removeEntry(this.name)}catch{}this._file=await n.getFileHandle(this.name,{create:!0}),this._syncHandle=await this._file.createSyncAccessHandle(),this._writeOffset=0,this._finalized=!1,this._totalEntries=0}catch{console.warn("[OPFSResultBuffer] OPFS not available, using memory-only mode"),this._syncHandle=null}}async appendMatches(e){if(this._finalized)throw new Error("Buffer already finalized");if(this._totalEntries+=e.length,!this._syncHandle){this._memBuffer.push(e.slice());return}this._memBuffer.push(e),this._memSize+=e.byteLength,this._memSize>=this.maxMemory&&await this._flush()}async _flush(){if(!this._syncHandle||this._memBuffer.length===0)return;let e=this._memBuffer.reduce((r,s)=>r+s.byteLength,0),t=new Uint8Array(e),n=0;for(let r of this._memBuffer)t.set(new Uint8Array(r.buffer,r.byteOffset,r.byteLength),n),n+=r.byteLength;this._syncHandle.write(t,{at:this._writeOffset}),this._writeOffset+=e,this._memBuffer=[],this._memSize=0}async finalize(){return this._finalized?this.stats():(await this._flush(),this._finalized=!0,this.stats())}stats(){return{totalEntries:this._totalEntries,totalBytes:this._writeOffset+this._memSize,onDisk:this._writeOffset,inMemory:this._memSize,entrySize:this._entrySize}}async*stream(e=16384){let t=e*this._entrySize,n=this._entrySize===8?BigUint64Array:Uint32Array;if(this._syncHandle&&this._writeOffset>0){let r=0;for(;r<this._writeOffset;){let s=this._writeOffset-r,i=Math.min(t,s),o=new ArrayBuffer(i),c=new Uint8Array(o);this._syncHandle.read(c,{at:r}),r+=i,yield new n(o)}}for(let r of this._memBuffer)yield r}async readAll(){let e=this._entrySize===8?BigUint64Array:Uint32Array,t=this._writeOffset+this._memBuffer.reduce((s,i)=>s+i.byteLength,0),n=new e(t/this._entrySize),r=0;for await(let s of this.stream())n.set(s,r),r+=s.length;return n}async close(e=!0){if(this._syncHandle&&(this._syncHandle.close(),this._syncHandle=null),e&&this._file)try{await(await(await navigator.storage.getDirectory()).getDirectoryHandle(st)).removeEntry(this.name)}catch{}this._memBuffer=[],this._memSize=0}};function D(a="temp"){let e=`${a}-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;return new ge(e)}var dn=`
struct Params {
    size: u32,
    num_partitions: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> partition_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> partition_counts: array<atomic<u32>>;

fn hash_partition(key: u32) -> u32 {
    // FNV-1a hash for partitioning
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

@compute @workgroup_size(256)
fn assign_partitions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let key = keys[idx];
    let part_id = hash_partition(key) % params.num_partitions;
    partition_ids[idx] = part_id;
    atomicAdd(&partition_counts[part_id], 1u);
}
`,mn=`
struct BuildParams { size: u32, capacity: u32 }
struct ProbeParams { left_size: u32, capacity: u32, max_matches: u32 }

@group(0) @binding(0) var<uniform> build_params: BuildParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>;

fn fnv_hash(key: u32) -> u32 {
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

@compute @workgroup_size(256)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= build_params.size) { return; }
    let key = keys[tid];
    var slot = fnv_hash(key) % build_params.capacity;
    for (var p = 0u; p < build_params.capacity; p++) {
        let idx = slot * 2u;
        let old = atomicCompareExchangeWeak(&hash_table[idx], 0xFFFFFFFFu, key);
        if (old.exchanged) {
            atomicStore(&hash_table[idx + 1u], tid);
            return;
        }
        slot = (slot + 1u) % build_params.capacity;
    }
}

@group(0) @binding(0) var<uniform> probe_params: ProbeParams;
@group(0) @binding(1) var<storage, read> left_keys: array<u32>;
@group(0) @binding(2) var<storage, read> probe_table: array<u32>;
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;
@group(0) @binding(4) var<storage, read_write> match_count: atomic<u32>;

@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let left_idx = gid.x;
    if (left_idx >= probe_params.left_size) { return; }
    let key = left_keys[left_idx];
    var slot = fnv_hash(key) % probe_params.capacity;
    for (var p = 0u; p < probe_params.capacity; p++) {
        let idx = slot * 2u;
        let stored = probe_table[idx];
        if (stored == 0xFFFFFFFFu) { return; }
        if (stored == key) {
            let right_idx = probe_table[idx + 1u];
            let out = atomicAdd(&match_count, 1u);
            if (out * 2u + 1u < probe_params.max_matches * 2u) {
                matches[out * 2u] = left_idx;
                matches[out * 2u + 1u] = right_idx;
            }
        }
        slot = (slot + 1u) % probe_params.capacity;
    }
}

struct InitParams { capacity: u32 }
@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> table_data: array<u32>;

@compute @workgroup_size(256)
fn clear_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= init_params.capacity * 2u) { return; }
    table_data[idx] = select(0u, 0xFFFFFFFFu, idx % 2u == 0u);
}
`,gn=256*1024*1024,pn=1e5,pe=class{constructor(){this.device=null,this.pipelines=new Map,this.available=!1,this._initPromise=null,this.bufferPool=null}async init(){return this._initPromise?this._initPromise:(this._initPromise=this._doInit(),this._initPromise)}isAvailable(){return this.available}async _doInit(){if(typeof navigator>"u"||!navigator.gpu)return console.log("[ChunkedGPUJoiner] WebGPU not available"),!1;try{let e=await navigator.gpu.requestAdapter();return e?(this.device=await e.requestDevice({requiredLimits:{maxStorageBufferBindingSize:256*1024*1024,maxBufferSize:256*1024*1024}}),await this._compileShaders(),this.bufferPool=M(this.device),this.available=!0,console.log("[ChunkedGPUJoiner] Initialized"),!0):!1}catch(e){return console.error("[ChunkedGPUJoiner] Init failed:",e),!1}}async _compileShaders(){let e=this.device.createShaderModule({code:dn});this.pipelines.set("assign_partitions",this.device.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:"assign_partitions"}}));let t=this.device.createShaderModule({code:mn});this.pipelines.set("build",this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"build"}})),this.pipelines.set("probe",this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"probe"}})),this.pipelines.set("clear_table",this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"clear_table"}}))}async hashJoin(e,t,n={},r,s){if(typeof n=="string")return this.hashJoinFlat(e,t,n,r,s||"INNER");let i=n,o=i.gpuMemoryBudget||gn,c=i.joinType||"INNER",l=await this._collectChunks(e),f=await this._collectChunks(t),u=l.keys.length+f.keys.length;return this._estimateMemory(l.keys.length,f.keys.length)<o&&u<pn?this._simpleJoin(l,f,c):this._partitionedJoin(l,f,o,c)}async _collectChunks(e){let t=[],n=[];for await(let c of e)t.push(c.keys),n.push(c.indices);let r=t.reduce((c,l)=>c+l.length,0),s=new Uint32Array(r),i=new Uint32Array(r),o=0;for(let c=0;c<t.length;c++)s.set(t[c],o),i.set(n[c],o),o+=t[c].length;return{keys:s,indices:i}}_estimateMemory(e,t){let n=t*4*2*4,r=(e+t)*4,s=t*10*2*4;return n+r+s}async _simpleJoin(e,t,n){let r=D("join");if(await r.init(),!this.available){let i=this._cpuHashJoin(e,t,n);return await r.appendMatches(i),await r.finalize(),r}let s=await this._gpuJoinPartition(e.keys,e.indices,t.keys,t.indices,n);return await r.appendMatches(s),await r.finalize(),r}async _partitionedJoin(e,t,n,r){let s=this._estimateMemory(e.keys.length,t.keys.length),i=Math.max(1,Math.ceil(s/n)*2);console.log(`[ChunkedGPUJoiner] Using ${i} partitions for ${e.keys.length} x ${t.keys.length} join`);let o=this._partitionData(e,i),c=this._partitionData(t,i),l=D("join");await l.init();for(let f=0;f<i;f++){let u=o[f],h=c[f];if(u.keys.length===0||h.keys.length===0)continue;console.log(`[ChunkedGPUJoiner] Partition ${f}: ${u.keys.length} x ${h.keys.length}`);let d=this.available?await this._gpuJoinPartition(u.keys,u.indices,h.keys,h.indices,r):this._cpuHashJoin(u,h,r);d.length>0&&await l.appendMatches(d)}return await l.finalize(),l}_partitionData(e,t){let n=Array.from({length:t},()=>({keys:[],indices:[]}));for(let r=0;r<e.keys.length;r++){let s=e.keys[r],i=this._hashPartition(s,t);n[i].keys.push(s),n[i].indices.push(e.indices[r])}return n.map(r=>({keys:new Uint32Array(r.keys),indices:new Uint32Array(r.indices)}))}_hashPartition(e,t){let n=2166136261;return n^=e&255,n=Math.imul(n,16777619),n^=e>>8&255,n=Math.imul(n,16777619),n^=e>>16&255,n=Math.imul(n,16777619),n^=e>>24&255,n=Math.imul(n,16777619),(n>>>0)%t}async _gpuJoinPartition(e,t,n,r,s){let i=this._nextPowerOf2(n.length*4),o=Math.max(e.length*10,1e5),c=this._createBuffer(n,GPUBufferUsage.STORAGE),l=this._createBuffer(e,GPUBufferUsage.STORAGE),f=this.device.createBuffer({size:i*2*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),u=this.device.createBuffer({size:o*2*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),h=this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),d=this.device.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});try{await this._clearHashTable(f,i),await this._buildHashTable(c,f,n.length,i),await this._probeHashTable(l,f,u,h,e.length,i,o);let m=this.device.createCommandEncoder();m.copyBufferToBuffer(h,0,d,0,4),this.device.queue.submit([m.finish()]),await d.mapAsync(GPUMapMode.READ);let g=new Uint32Array(d.getMappedRange())[0];if(d.unmap(),g===0)return new Uint32Array(0);let b=Math.min(g,o),w=this.device.createBuffer({size:b*2*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),p=this.device.createCommandEncoder();p.copyBufferToBuffer(u,0,w,0,b*2*4),this.device.queue.submit([p.finish()]),await w.mapAsync(GPUMapMode.READ);let y=new Uint32Array(w.getMappedRange().slice(0));w.unmap(),w.destroy();let _=new Uint32Array(b*2);for(let x=0;x<b;x++){let v=y[x*2],A=y[x*2+1];_[x*2]=t[v],_[x*2+1]=r[A]}return _}finally{c.destroy(),l.destroy(),f.destroy(),u.destroy(),h.destroy(),d.destroy()}}async _clearHashTable(e,t){let n=this.device.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(n,0,new Uint32Array([t]));let r=this.pipelines.get("clear_table"),s=this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:e}}]}),i=this.device.createCommandEncoder(),o=i.beginComputePass();o.setPipeline(r),o.setBindGroup(0,s),o.dispatchWorkgroups(Math.ceil(t*2/256)),o.end(),this.device.queue.submit([i.finish()]),await this.device.queue.onSubmittedWorkDone(),n.destroy()}async _buildHashTable(e,t,n,r){let s=this.device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(s,0,new Uint32Array([n,r]));let i=this.pipelines.get("build"),o=this.device.createBindGroup({layout:i.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:e}},{binding:2,resource:{buffer:t}}]}),c=this.device.createCommandEncoder(),l=c.beginComputePass();l.setPipeline(i),l.setBindGroup(0,o),l.dispatchWorkgroups(Math.ceil(n/256)),l.end(),this.device.queue.submit([c.finish()]),await this.device.queue.onSubmittedWorkDone(),s.destroy()}async _probeHashTable(e,t,n,r,s,i,o){this.device.queue.writeBuffer(r,0,new Uint32Array([0]));let c=this.device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(c,0,new Uint32Array([s,i,o]));let l=this.pipelines.get("probe"),f=this.device.createBindGroup({layout:l.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:c}},{binding:1,resource:{buffer:e}},{binding:2,resource:{buffer:t}},{binding:3,resource:{buffer:n}},{binding:4,resource:{buffer:r}}]}),u=this.device.createCommandEncoder(),h=u.beginComputePass();h.setPipeline(l),h.setBindGroup(0,f),h.dispatchWorkgroups(Math.ceil(s/256)),h.end(),this.device.queue.submit([u.finish()]),await this.device.queue.onSubmittedWorkDone(),c.destroy()}_createBuffer(e,t){let n=this.device.createBuffer({size:e.byteLength,usage:t|GPUBufferUsage.COPY_DST});return this.device.queue.writeBuffer(n,0,e),n}_nextPowerOf2(e){let t=1;for(;t<e;)t*=2;return t}_cpuHashJoin(e,t,n){let r=new Map;for(let i=0;i<t.keys.length;i++){let o=t.keys[i];r.has(o)||r.set(o,[]),r.get(o).push(t.indices[i])}let s=[];for(let i=0;i<e.keys.length;i++){let o=e.keys[i],c=r.get(o);if(c)for(let l of c)s.push(e.indices[i],l)}return new Uint32Array(s)}async hashJoinFlat(e,t,n,r,s="INNER",i={}){let o=this._extractKeysFromRows(e,n),c=this._extractKeysFromRows(t,r),l=new Uint32Array(e.length).map((y,_)=>_),f=new Uint32Array(t.length).map((y,_)=>_),u=this;async function*h(){yield{keys:o,indices:l}}async function*d(){yield{keys:c,indices:f}}let m=await this.hashJoin(h(),d(),{joinType:s,...i}),g=await m.readAll();await m.close(!0);let b=g.length/2,w=new Uint32Array(b),p=new Uint32Array(b);for(let y=0;y<b;y++)w[y]=g[y*2],p[y]=g[y*2+1];return{leftIndices:w,rightIndices:p,matchCount:b}}_extractKeysFromRows(e,t){let n=new Uint32Array(e.length);for(let r=0;r<e.length;r++){let s=e[r][t];n[r]=typeof s=="number"?s:this._hashStringKey(String(s))}return n}_hashStringKey(e){let t=0;for(let n=0;n<e.length;n++)t=(t<<5)-t+e.charCodeAt(n),t|=0;return t>>>0}},ke=null;function it(){return ke||(ke=new pe),ke}var ot=it;var wn=`
struct Params {
    size: u32,
    stage: u32,
    step: u32,
    descending: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(256)
fn bitonic_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let size = params.size;
    let stage = params.stage;
    let step = params.step;
    let desc = params.descending == 1u;

    let half_step = 1u << step;
    let full_step = half_step << 1u;

    let block = tid / half_step;
    let idx = tid % half_step;
    let i = block * full_step + idx;
    let j = i + half_step;

    if (j >= size) { return; }

    let direction = ((i >> (stage + 1u)) & 1u) == 0u;
    let ascending = direction != desc;

    let ki = keys[i];
    let kj = keys[j];
    let should_swap = select(ki > kj, ki < kj, ascending);

    if (should_swap) {
        keys[i] = kj;
        keys[j] = ki;
        let ii = indices[i];
        indices[i] = indices[j];
        indices[j] = ii;
    }
}
`,at=256*1024*1024,yn=1e4,bn=8,we=class{constructor(){this.device=null,this.pipelines=new Map,this.available=!1,this._initPromise=null,this.bufferPool=null}async init(){return this._initPromise?this._initPromise:(this._initPromise=this._doInit(),this._initPromise)}async _doInit(){if(typeof navigator>"u"||!navigator.gpu)return console.log("[ChunkedGPUSorter] WebGPU not available"),!1;try{let e=await navigator.gpu.requestAdapter();return e?(this.device=await e.requestDevice({requiredLimits:{maxStorageBufferBindingSize:256*1024*1024,maxBufferSize:256*1024*1024}}),await this._compileShaders(),this.bufferPool=M(this.device),this.available=!0,console.log("[ChunkedGPUSorter] Initialized"),!0):!1}catch(e){return console.error("[ChunkedGPUSorter] Init failed:",e),!1}}async _compileShaders(){let e=this.device.createShaderModule({code:wn});this.pipelines.set("bitonic_step",this.device.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:"bitonic_step"}}))}calculateChunkSize(e=at){let t=e*.4;return Math.floor(t/bn)}async sortWithLimit(e,t,n=!0,r={}){let s=t,i={keys:new Float32Array(0),indices:new Uint32Array(0)},o=0;for await(let c of e){o+=c.keys.length;let l=this._combineChunks(i,c),f=this._cpuSort(l.keys,l.indices,n),u=Math.min(s,f.keys.length);i={keys:f.keys.slice(0,u),indices:f.indices.slice(0,u)}}return console.log(`[ChunkedGPUSorter] sortWithLimit: ${o} elements \u2192 top ${i.keys.length}`),i}async fullSort(e,t=!0,n={}){let r=n.gpuMemoryBudget||at,s=this.calculateChunkSize(r),i=[],o=0;for await(let l of e){console.log(`[ChunkedGPUSorter] Sorting chunk ${o}: ${l.keys.length} elements`);let f=await this._sortChunk(l.keys,l.indices,t),u=D(`sort-chunk-${o}`);await u.init(4);let h=new Float32Array(f.keys.length*2);for(let d=0;d<f.keys.length;d++)h[d*2]=f.keys[d],h[d*2+1]=new DataView(new Uint32Array([f.indices[d]]).buffer).getFloat32(0,!0);await u.appendMatches(new Uint32Array(h.buffer)),await u.finalize(),i.push({buffer:u,length:f.keys.length}),o++}console.log(`[ChunkedGPUSorter] Merging ${i.length} sorted chunks`);let c=D("sort-result");await c.init(4),await this._kWayMerge(i,c,t);for(let l of i)await l.buffer.close(!0);return await c.finalize(),c}async _sortChunk(e,t,n){let r=e.length;return!this.available||r<yn?this._cpuSort(e,t,n):this._gpuBitonicSort(e,t,n)}async _gpuBitonicSort(e,t,n){let r=e.length,s=this._nextPowerOf2(r),i=new Float32Array(s),o=new Uint32Array(s);i.set(e),o.set(t);let c=n?-1/0:1/0;for(let h=r;h<s;h++)i[h]=c,o[h]=4294967295;let l=this.device.createBuffer({size:i.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});this.device.queue.writeBuffer(l,0,i);let f=this.device.createBuffer({size:o.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});this.device.queue.writeBuffer(f,0,o);let u=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});try{let h=this.pipelines.get("bitonic_step"),d=Math.ceil(Math.log2(s));for(let y=0;y<d;y++)for(let _=y;_>=0;_--){this.device.queue.writeBuffer(u,0,new Uint32Array([s,y,_,n?1:0]));let x=this.device.createBindGroup({layout:h.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:u}},{binding:1,resource:{buffer:l}},{binding:2,resource:{buffer:f}}]}),v=this.device.createCommandEncoder(),A=v.beginComputePass();A.setPipeline(h),A.setBindGroup(0,x),A.dispatchWorkgroups(Math.ceil(s/2/256)),A.end(),this.device.queue.submit([v.finish()])}await this.device.queue.onSubmittedWorkDone();let m=this.device.createBuffer({size:r*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),g=this.device.createBuffer({size:r*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),b=this.device.createCommandEncoder();b.copyBufferToBuffer(l,0,m,0,r*4),b.copyBufferToBuffer(f,0,g,0,r*4),this.device.queue.submit([b.finish()]),await m.mapAsync(GPUMapMode.READ),await g.mapAsync(GPUMapMode.READ);let w=new Float32Array(m.getMappedRange().slice(0)),p=new Uint32Array(g.getMappedRange().slice(0));return m.unmap(),g.unmap(),m.destroy(),g.destroy(),{keys:w,indices:p}}finally{l.destroy(),f.destroy(),u.destroy()}}_cpuSort(e,t,n){let r=[];for(let s=0;s<e.length;s++)r.push({key:e[s],index:t[s]});return r.sort((s,i)=>n?i.key-s.key:s.key-i.key),{keys:new Float32Array(r.map(s=>s.key)),indices:new Uint32Array(r.map(s=>s.index))}}_combineChunks(e,t){let n=new Float32Array(e.keys.length+t.keys.length),r=new Uint32Array(e.indices.length+t.indices.length);return n.set(e.keys),n.set(t.keys,e.keys.length),r.set(e.indices),r.set(t.indices,e.indices.length),{keys:n,indices:r}}async _kWayMerge(e,t,n){let r=[];for(let s of e){let i=await s.buffer.readAll(),o=i.length/2;r.push({data:i,length:o,pos:0})}for(;;){let s=-1,i=n?-1/0:1/0;for(let f=0;f<r.length;f++){let u=r[f];if(u.pos>=u.length)continue;let d=new DataView(u.data.buffer,u.data.byteOffset).getFloat32(u.pos*8,!0);(n?d>i:d<i)&&(i=d,s=f)}if(s===-1)break;let o=r[s],l=new DataView(o.data.buffer,o.data.byteOffset).getUint32(o.pos*8+4,!0);await t.appendMatches(new Uint32Array([l])),o.pos++}}_nextPowerOf2(e){let t=1;for(;t<e;)t*=2;return t}async sort(e,t=!0){let n=new Uint32Array(e.length).map((s,i)=>i);return(await this._sortChunk(e,n,!t)).indices}isAvailable(){return this.available}},Be=null;function ct(){return Be||(Be=new we),Be}var lt=ct;var _n=`
struct Params {
    size: u32,
    capacity: u32,
    agg_type: u32,  // 0=COUNT, 1=SUM, 2=MIN, 3=MAX
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> hash_table: array<atomic<u32>>;  // key, count, sum_bits_lo, sum_bits_hi
@group(0) @binding(4) var<storage, read_write> group_count: atomic<u32>;

fn fnv_hash(key: u32) -> u32 {
    var h = 2166136261u;
    h ^= (key & 0xFFu); h *= 16777619u;
    h ^= ((key >> 8u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 16u) & 0xFFu); h *= 16777619u;
    h ^= ((key >> 24u) & 0xFFu); h *= 16777619u;
    return h;
}

// Convert f32 to sortable u32 for atomic MIN/MAX
fn f32_to_sortable(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

fn sortable_to_f32(s: u32) -> f32 {
    let mask = select(0x80000000u, 0xFFFFFFFFu, (s & 0x80000000u) == 0u);
    return bitcast<f32>(s ^ mask);
}

@compute @workgroup_size(256)
fn aggregate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let key = keys[idx];
    let val = values[idx];
    var slot = fnv_hash(key) % params.capacity;

    // Find or create slot for this key
    for (var p = 0u; p < params.capacity; p++) {
        let base = slot * 4u;

        // Try to claim this slot
        let old_key = atomicCompareExchangeWeak(&hash_table[base], 0xFFFFFFFFu, key);

        if (old_key.exchanged || old_key.old_value == key) {
            // This slot is ours
            if (old_key.exchanged) {
                atomicAdd(&group_count, 1u);
            }

            // Update aggregate
            if (params.agg_type == 0u) {
                // COUNT
                atomicAdd(&hash_table[base + 1u], 1u);
            } else if (params.agg_type == 1u) {
                // SUM: Use float accumulation via atomic add on sortable bits
                // Note: This is approximate for floats, consider f64 split for precision
                let val_bits = bitcast<u32>(val);
                atomicAdd(&hash_table[base + 2u], val_bits);
            } else if (params.agg_type == 2u) {
                // MIN
                let sortable = f32_to_sortable(val);
                atomicMin(&hash_table[base + 2u], sortable);
            } else if (params.agg_type == 3u) {
                // MAX
                let sortable = f32_to_sortable(val);
                atomicMax(&hash_table[base + 2u], sortable);
            }
            return;
        }

        slot = (slot + 1u) % params.capacity;
    }
}

struct InitParams { capacity: u32 }
@group(0) @binding(0) var<uniform> init_params: InitParams;
@group(0) @binding(1) var<storage, read_write> table_data: array<u32>;

@compute @workgroup_size(256)
fn clear_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let entry_size = 4u;  // key, count, agg_lo, agg_hi
    if (idx >= init_params.capacity * entry_size) { return; }

    let entry_idx = idx / entry_size;
    let field_idx = idx % entry_size;

    if (field_idx == 0u) {
        // Key: 0xFFFFFFFF = empty
        table_data[idx] = 0xFFFFFFFFu;
    } else if (field_idx == 2u) {
        // Aggregate field: init depends on agg type
        // For MIN: init to MAX_FLOAT sortable
        // For MAX: init to MIN_FLOAT sortable
        // For SUM/COUNT: init to 0
        table_data[idx] = 0u;
    } else {
        table_data[idx] = 0u;
    }
}
`,xn=0,Re=1,vn=2,An=3,In={COUNT:xn,SUM:Re,MIN:vn,MAX:An,AVG:Re},Sn=256*1024*1024,Fn=1e4,ye=class{constructor(){this.device=null,this.pipelines=new Map,this.available=!1,this._initPromise=null,this.bufferPool=null}async init(){return this._initPromise?this._initPromise:(this._initPromise=this._doInit(),this._initPromise)}isAvailable(){return this.available}async _doInit(){if(typeof navigator>"u"||!navigator.gpu)return console.log("[ChunkedGPUGrouper] WebGPU not available"),!1;try{let e=await navigator.gpu.requestAdapter();return e?(this.device=await e.requestDevice({requiredLimits:{maxStorageBufferBindingSize:256*1024*1024,maxBufferSize:256*1024*1024}}),await this._compileShaders(),this.bufferPool=M(this.device),this.available=!0,console.log("[ChunkedGPUGrouper] Initialized"),!0):!1}catch(e){return console.error("[ChunkedGPUGrouper] Init failed:",e),!1}}async _compileShaders(){let e=this.device.createShaderModule({code:_n});this.pipelines.set("aggregate",this.device.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:"aggregate"}})),this.pipelines.set("clear_table",this.device.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:"clear_table"}}))}async groupBy(e,t,n={}){if(e instanceof Uint32Array)return this.groupByFlat(e);let r=n.gpuMemoryBudget||Sn,s=n.estimatedGroups||1e5,i=t==="AVG",o=new Map,c=0;for await(let f of e){c+=f.keys.length;let u=await this._processChunk(f.keys,f.values,t,r);this._mergePartials(o,u,t)}console.log(`[ChunkedGPUGrouper] Processed ${c} rows into ${o.size} groups`);let l=new Map;for(let[f,u]of o){let h;switch(t){case"COUNT":h=u.count;break;case"SUM":h=u.sum;break;case"MIN":h=u.min;break;case"MAX":h=u.max;break;case"AVG":h=u.count>0?u.sum/u.count:0;break}l.set(f,h)}return l}async _processChunk(e,t,n,r){let s=e.length;return!this.available||s<Fn?this._cpuAggregate(e,t,n):this._gpuAggregate(e,t,n)}async _gpuAggregate(e,t,n){let r=e.length,s=this._nextPowerOf2(Math.max(r,1024)*2),i=In[n]??Re,o=this._createBuffer(e,GPUBufferUsage.STORAGE),c=t instanceof Float32Array?t:new Float32Array(t),l=this._createBuffer(c,GPUBufferUsage.STORAGE),f=this.device.createBuffer({size:s*4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),u=this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});try{await this._clearHashTable(f,s),this.device.queue.writeBuffer(u,0,new Uint32Array([0]));let h=this.device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(h,0,new Uint32Array([r,s,i]));let d=this.pipelines.get("aggregate"),m=this.device.createBindGroup({layout:d.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:u}}]}),g=this.device.createCommandEncoder(),b=g.beginComputePass();b.setPipeline(d),b.setBindGroup(0,m),b.dispatchWorkgroups(Math.ceil(r/256)),b.end(),this.device.queue.submit([g.finish()]),await this.device.queue.onSubmittedWorkDone(),h.destroy();let w=this.device.createBuffer({size:s*4*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),p=this.device.createCommandEncoder();p.copyBufferToBuffer(f,0,w,0,s*4*4),this.device.queue.submit([p.finish()]),await w.mapAsync(GPUMapMode.READ);let y=new Uint32Array(w.getMappedRange().slice(0));w.unmap(),w.destroy();let _=new Map;for(let x=0;x<s;x++){let v=x*4,A=y[v];if(A===4294967295)continue;let I=y[v+1],S=y[v+2],U;n==="COUNT"?U=I:n==="MIN"||n==="MAX"?U=this._sortableToF32(S):U=this._bitsToFloat(S),_.set(A,{count:I||1,sum:n==="SUM"||n==="AVG"?U:0,min:n==="MIN"?U:1/0,max:n==="MAX"?U:-1/0})}return _}finally{o.destroy(),l.destroy(),f.destroy(),u.destroy()}}async _clearHashTable(e,t){let n=this.device.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(n,0,new Uint32Array([t]));let r=this.pipelines.get("clear_table"),s=this.device.createBindGroup({layout:r.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:e}}]}),i=this.device.createCommandEncoder(),o=i.beginComputePass();o.setPipeline(r),o.setBindGroup(0,s),o.dispatchWorkgroups(Math.ceil(t*4/256)),o.end(),this.device.queue.submit([i.finish()]),await this.device.queue.onSubmittedWorkDone(),n.destroy()}_mergePartials(e,t,n){for(let[r,s]of t){e.has(r)||e.set(r,{count:0,sum:0,min:1/0,max:-1/0});let i=e.get(r);i.count+=s.count,i.sum+=s.sum,i.min=Math.min(i.min,s.min),i.max=Math.max(i.max,s.max)}}_cpuAggregate(e,t,n){let r=new Map;for(let s=0;s<e.length;s++){let i=e[s],o=t[s];r.has(i)||r.set(i,{count:0,sum:0,min:1/0,max:-1/0});let c=r.get(i);c.count++,c.sum+=o,c.min=Math.min(c.min,o),c.max=Math.max(c.max,o)}return r}_createBuffer(e,t){let n=this.device.createBuffer({size:e.byteLength,usage:t|GPUBufferUsage.COPY_DST});return this.device.queue.writeBuffer(n,0,e),n}_nextPowerOf2(e){let t=1;for(;t<e;)t*=2;return t}_f32ToSortable(e){let t=new DataView(new ArrayBuffer(4));t.setFloat32(0,e,!0);let n=t.getUint32(0,!0),r=n&2147483648?4294967295:2147483648;return n^r}_sortableToF32(e){let t=e&2147483648?2147483648:4294967295,n=e^t,r=new DataView(new ArrayBuffer(4));return r.setUint32(0,n,!0),r.getFloat32(0,!0)}_bitsToFloat(e){let t=new DataView(new ArrayBuffer(4));return t.setUint32(0,e,!0),t.getFloat32(0,!0)}async groupByFlat(e){let t=new Map,n=new Uint32Array(e.length),r=0;for(let s=0;s<e.length;s++){let i=e[s];t.has(i)||t.set(i,r++),n[s]=t.get(i)}return{groupIds:n,numGroups:t.size}}async groupAggregateFlat(e,t,n,r){let s=new Float32Array(n),i=new Uint32Array(n);r==="MIN"?s.fill(1/0):r==="MAX"&&s.fill(-1/0);for(let o=0;o<e.length;o++){let c=t[o],l=e[o];switch(i[c]++,r){case"COUNT":s[c]++;break;case"SUM":case"AVG":s[c]+=l;break;case"MIN":s[c]=Math.min(s[c],l);break;case"MAX":s[c]=Math.max(s[c],l);break}}if(r==="AVG")for(let o=0;o<n;o++)i[o]>0&&(s[o]/=i[o]);return s}async groupBySimple(e){return this.groupByFlat(e)}async groupAggregate(e,t,n,r){return this.groupAggregateFlat(e,t,n,r)}},Me=null;function ut(){return Me||(Me=new ye),Me}var ft=ut;var R={COSINE:0,L2:1,DOT_PRODUCT:2},Un=`
struct Params {
    dim: u32,
    num_vectors: u32,
    metric: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

var<workgroup> shared_query: array<f32, 512>;

@compute @workgroup_size(256)
fn compute_distances(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let vec_idx = gid.x;
    if (vec_idx >= params.num_vectors) { return; }

    let dim = params.dim;
    let tid = lid.x;

    // Load query into shared memory
    for (var i = tid; i < dim && i < 512u; i += 256u) {
        shared_query[i] = query[i];
    }
    workgroupBarrier();

    let vec_offset = vec_idx * dim;
    var result: f32 = 0.0;

    if (params.metric == 1u) {
        // L2 distance
        var sum: f32 = 0.0;
        for (var i = 0u; i < dim; i++) {
            let d = shared_query[i] - vectors[vec_offset + i];
            sum += d * d;
        }
        result = sqrt(sum);
    } else {
        // Cosine / Dot product
        var dot: f32 = 0.0;
        for (var i = 0u; i < dim; i++) {
            dot += shared_query[i] * vectors[vec_offset + i];
        }
        result = dot;
    }

    distances[vec_idx] = result;
}
`,Pn=`
struct Params {
    total_candidates: u32,
    k: u32,
    descending: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32>;

// Simple insertion sort for small K
@compute @workgroup_size(1)
fn merge_topk() {
    let k = params.k;
    let n = params.total_candidates;
    let desc = params.descending == 1u;

    // Initialize output with worst values
    let sentinel = select(3.4028235e+38, -3.4028235e+38, desc);
    for (var i = 0u; i < k; i++) {
        out_scores[i] = sentinel;
        out_indices[i] = 0xFFFFFFFFu;
    }

    // Insertion sort each candidate into top-K
    for (var i = 0u; i < n; i++) {
        let score = scores[i];
        let idx = indices[i];

        // Find insertion position
        var pos = k;
        for (var j = 0u; j < k; j++) {
            let better = select(score < out_scores[j], score > out_scores[j], desc);
            if (better) {
                pos = j;
                break;
            }
        }

        // Insert if in top-K
        if (pos < k) {
            // Shift elements down
            for (var j = k - 1u; j > pos; j--) {
                out_scores[j] = out_scores[j - 1u];
                out_indices[j] = out_indices[j - 1u];
            }
            out_scores[pos] = score;
            out_indices[pos] = idx;
        }
    }
}
`,ht=256*1024*1024,X=4,be=class{constructor(){this.device=null,this.pipelines=new Map,this.available=!1,this._initPromise=null,this.bufferPool=null}async init(){return this._initPromise?this._initPromise:(this._initPromise=this._doInit(),this._initPromise)}async _doInit(){if(typeof navigator>"u"||!navigator.gpu)return console.log("[ChunkedGPUVectorSearch] WebGPU not available"),!1;try{let e=await navigator.gpu.requestAdapter();return e?(this.device=await e.requestDevice({requiredLimits:{maxStorageBufferBindingSize:256*1024*1024,maxBufferSize:256*1024*1024}}),await this._compileShaders(),this.bufferPool=M(this.device),this.available=!0,console.log("[ChunkedGPUVectorSearch] Initialized"),!0):!1}catch(e){return console.error("[ChunkedGPUVectorSearch] Init failed:",e),!1}}async _compileShaders(){let e=this.device.createShaderModule({code:Un});this.pipelines.set("compute_distances",this.device.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:"compute_distances"}}));let t=this.device.createShaderModule({code:Pn});this.pipelines.set("merge_topk",this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"merge_topk"}}))}calculateChunkSize(e,t=ht){let n=(e+1)*X,r=e*X+t*.2,s=t-r,i=Math.floor(s/n);return Math.max(1e3,Math.min(i,5e5))}async search(e,t,n={},r={}){if(typeof n=="number"){let u=n;return this.searchFlat(e,t,{k:u,...r})}if(Array.isArray(t))return this.searchFlat(e,t,n);let s=n,i=s.k||10,o=s.metric??R.COSINE,c=o===R.COSINE||o===R.DOT_PRODUCT,l=[],f=0;for await(let u of t){let{vectors:h,startIndex:d}=u,m=h.length;console.log(`[ChunkedGPUVectorSearch] Processing chunk: ${m} vectors, startIndex=${d}`);let g=await this._computeChunkDistances(e,h,o),b=this._cpuTopK(g,d,i,c);l.push(b),f+=m}return console.log(`[ChunkedGPUVectorSearch] Processed ${f} total vectors in ${l.length} chunks`),this._mergeTopK(l,i,c)}async searchFlat(e,t,n={}){let r=n.k||10,s=n.metric??R.COSINE,i=n.gpuMemoryBudget||ht,o=e.length,c=this.calculateChunkSize(o,i);console.log(`[ChunkedGPUVectorSearch] Chunk size: ${c} vectors (dim=${o})`);let l=this;async function*f(){for(let u=0;u<t.length;u+=c){let h=Math.min(u+c,t.length);yield{vectors:t.slice(u,h),startIndex:u}}}return this.search(e,f(),{k:r,metric:s,gpuMemoryBudget:i})}async _computeChunkDistances(e,t,n){let r=e.length,s=t.length;if(!this.available||s<1e3)return this._cpuDistances(e,t,n);let i=new Float32Array(s*r);for(let u=0;u<s;u++)i.set(t[u],u*r);let o=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(o,0,new Uint32Array([r,s,n,0]));let c=this.device.createBuffer({size:e.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(c,0,e);let l=this.device.createBuffer({size:i.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(l,0,i);let f=this.device.createBuffer({size:s*X,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});try{let u=this.pipelines.get("compute_distances"),h=this.device.createBindGroup({layout:u.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:f}}]}),d=this.device.createCommandEncoder(),m=d.beginComputePass();m.setPipeline(u),m.setBindGroup(0,h),m.dispatchWorkgroups(Math.ceil(s/256)),m.end(),this.device.queue.submit([d.finish()]),await this.device.queue.onSubmittedWorkDone();let g=this.device.createBuffer({size:s*X,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),b=this.device.createCommandEncoder();b.copyBufferToBuffer(f,0,g,0,s*X),this.device.queue.submit([b.finish()]),await g.mapAsync(GPUMapMode.READ);let w=new Float32Array(g.getMappedRange().slice(0));return g.unmap(),g.destroy(),w}finally{o.destroy(),c.destroy(),l.destroy(),f.destroy()}}_cpuDistances(e,t,n){let r=new Float32Array(t.length),s=e.length;for(let i=0;i<t.length;i++){let o=t[i];if(n===R.L2){let c=0;for(let l=0;l<s;l++){let f=e[l]-o[l];c+=f*f}r[i]=Math.sqrt(c)}else{let c=0;for(let l=0;l<s;l++)c+=e[l]*o[l];r[i]=c}}return r}_cpuTopK(e,t,n,r){let s=[];for(let o=0;o<e.length;o++)s.push({index:t+o,score:e[o]});s.sort((o,c)=>r?c.score-o.score:o.score-c.score);let i=s.slice(0,Math.min(n,s.length));return{indices:new Uint32Array(i.map(o=>o.index)),scores:new Float32Array(i.map(o=>o.score))}}_mergeTopK(e,t,n){if(e.length===0)return{indices:new Uint32Array(0),scores:new Float32Array(0)};if(e.length===1)return e[0];let r=[];for(let i of e)for(let o=0;o<i.indices.length;o++)r.push({index:i.indices[o],score:i.scores[o]});r.sort((i,o)=>n?o.score-i.score:i.score-o.score);let s=r.slice(0,Math.min(t,r.length));return{indices:new Uint32Array(s.map(i=>i.index)),scores:new Float32Array(s.map(i=>i.score))}}isAvailable(){return this.available}async topK(e,t=null,n=10,r=!0){let s=e.length;if(!t){t=new Uint32Array(s);for(let i=0;i<s;i++)t[i]=i}return this._cpuTopKWithIndices(e,t,n,r)}_cpuTopKWithIndices(e,t,n,r){let s=[];for(let o=0;o<e.length;o++)s.push({index:t[o],score:e[o]});s.sort((o,c)=>r?c.score-o.score:o.score-c.score);let i=s.slice(0,Math.min(n,s.length));return{indices:new Uint32Array(i.map(o=>o.index)),scores:new Float32Array(i.map(o=>o.score))}}async computeDistances(e,t,n=1,r=R.COSINE){if(n===1)return this._computeChunkDistances(e,t,r);let s=e.length/n,i=t.length,o=new Float32Array(n*i);for(let c=0;c<n;c++){let l=c*s,f=e.slice(l,l+s),u=await this._computeChunkDistances(f,t,r);o.set(u,c*i)}return o}async searchSimple(e,t,n=10,r={}){let{metric:s=R.COSINE}=r,i=await this.computeDistances(e,t,1,s),o=s===R.COSINE||s===R.DOT_PRODUCT;return await this.topK(i,null,n,o)}},Ee=null;function dt(){return Ee||(Ee=new be),Ee}var Z=dt;var ee=class{constructor(e,t){this.lanceql=e,this.wasm=e.wasm,this.memory=e.memory;let n=new Uint8Array(t);if(this.dataPtr=this.wasm.alloc(n.length),!this.dataPtr)throw new Error("Failed to allocate memory for Lance file");if(this.dataLen=n.length,new Uint8Array(this.memory.buffer).set(n,this.dataPtr),this.wasm.openFile(this.dataPtr,this.dataLen)===0)throw this.wasm.free(this.dataPtr,this.dataLen),new Error("Failed to open Lance file")}close(){this.wasm.closeFile(),this.dataPtr&&(this.wasm.free(this.dataPtr,this.dataLen),this.dataPtr=null)}get numColumns(){return this.wasm.getNumColumns()}getRowCount(e){return this.wasm.getRowCount(e)}getColumnDebugInfo(e){return{offset:this.wasm.getColumnBufferOffset(e),size:this.wasm.getColumnBufferSize(e),rows:this.wasm.getRowCount(e)}}readInt64Column(e){let t=Number(this.getRowCount(e));if(t===0)return new BigInt64Array(0);let n=this.wasm.allocInt64Buffer(t);if(!n)throw new Error("Failed to allocate int64 buffer");try{let r=this.wasm.readInt64Column(e,n,t),s=new BigInt64Array(r),i=new BigInt64Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.freeInt64Buffer(n,t)}}readFloat64Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Float64Array(0);let n=this.wasm.allocFloat64Buffer(t);if(!n)throw new Error("Failed to allocate float64 buffer");try{let r=this.wasm.readFloat64Column(e,n,t),s=new Float64Array(r),i=new Float64Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.freeFloat64Buffer(n,t)}}readInt32Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Int32Array(0);let n=this.wasm.allocInt32Buffer(t);if(!n)throw new Error("Failed to allocate int32 buffer");try{let r=this.wasm.readInt32Column(e,n,t),s=new Int32Array(r),i=new Int32Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t*4)}}readInt16Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Int16Array(0);let n=this.wasm.allocInt16Buffer(t);if(!n)throw new Error("Failed to allocate int16 buffer");try{let r=this.wasm.readInt16Column(e,n,t),s=new Int16Array(r),i=new Int16Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t*2)}}readInt8Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Int8Array(0);let n=this.wasm.allocInt8Buffer(t);if(!n)throw new Error("Failed to allocate int8 buffer");try{let r=this.wasm.readInt8Column(e,n,t),s=new Int8Array(r),i=new Int8Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t)}}readUint64Column(e){let t=Number(this.getRowCount(e));if(t===0)return new BigUint64Array(0);let n=this.wasm.allocUint64Buffer(t);if(!n)throw new Error("Failed to allocate uint64 buffer");try{let r=this.wasm.readUint64Column(e,n,t),s=new BigUint64Array(r),i=new BigUint64Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t*8)}}readUint32Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Uint32Array(0);let n=this.wasm.allocIndexBuffer(t);if(!n)throw new Error("Failed to allocate uint32 buffer");try{let r=this.wasm.readUint32Column(e,n,t),s=new Uint32Array(r),i=new Uint32Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t*4)}}readUint16Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Uint16Array(0);let n=this.wasm.allocUint16Buffer(t);if(!n)throw new Error("Failed to allocate uint16 buffer");try{let r=this.wasm.readUint16Column(e,n,t),s=new Uint16Array(r),i=new Uint16Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t*2)}}readUint8Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Uint8Array(0);let n=this.wasm.allocStringBuffer(t);if(!n)throw new Error("Failed to allocate uint8 buffer");try{let r=this.wasm.readUint8Column(e,n,t),s=new Uint8Array(r),i=new Uint8Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t)}}readFloat32Column(e){let t=Number(this.getRowCount(e));if(t===0)return new Float32Array(0);let n=this.wasm.allocFloat32Buffer(t);if(!n)throw new Error("Failed to allocate float32 buffer");try{let r=this.wasm.readFloat32Column(e,n,t),s=new Float32Array(r),i=new Float32Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t*4)}}readBoolColumn(e){let t=Number(this.getRowCount(e));if(t===0)return new Uint8Array(0);let n=this.wasm.allocStringBuffer(t);if(!n)throw new Error("Failed to allocate bool buffer");try{let r=this.wasm.readBoolColumn(e,n,t),s=new Uint8Array(r),i=new Uint8Array(this.memory.buffer,n,r);return s.set(i),s}finally{this.wasm.free(n,t)}}readInt32AtIndices(e,t){if(t.length===0)return new Int32Array(0);let n=this.wasm.allocIndexBuffer(t.length);if(!n)throw new Error("Failed to allocate index buffer");let r=this.wasm.allocInt32Buffer(t.length);if(!r)throw this.wasm.free(n,t.length*4),new Error("Failed to allocate output buffer");try{new Uint32Array(this.memory.buffer,n,t.length).set(t);let s=this.wasm.readInt32AtIndices(e,n,t.length,r),i=new Int32Array(s),o=new Int32Array(this.memory.buffer,r,s);return i.set(o),i}finally{this.wasm.free(n,t.length*4),this.wasm.free(r,t.length*4)}}readFloat32AtIndices(e,t){if(t.length===0)return new Float32Array(0);let n=this.wasm.allocIndexBuffer(t.length);if(!n)throw new Error("Failed to allocate index buffer");let r=this.wasm.allocFloat32Buffer(t.length);if(!r)throw this.wasm.free(n,t.length*4),new Error("Failed to allocate output buffer");try{new Uint32Array(this.memory.buffer,n,t.length).set(t);let s=this.wasm.readFloat32AtIndices(e,n,t.length,r),i=new Float32Array(s),o=new Float32Array(this.memory.buffer,r,s);return i.set(o),i}finally{this.wasm.free(n,t.length*4),this.wasm.free(r,t.length*4)}}readUint8AtIndices(e,t){if(t.length===0)return new Uint8Array(0);let n=this.wasm.allocIndexBuffer(t.length);if(!n)throw new Error("Failed to allocate index buffer");let r=this.wasm.allocStringBuffer(t.length);if(!r)throw this.wasm.free(n,t.length*4),new Error("Failed to allocate output buffer");try{new Uint32Array(this.memory.buffer,n,t.length).set(t);let s=this.wasm.readUint8AtIndices(e,n,t.length,r),i=new Uint8Array(s),o=new Uint8Array(this.memory.buffer,r,s);return i.set(o),i}finally{this.wasm.free(n,t.length*4),this.wasm.free(r,t.length)}}readBoolAtIndices(e,t){if(t.length===0)return new Uint8Array(0);let n=this.wasm.allocIndexBuffer(t.length);if(!n)throw new Error("Failed to allocate index buffer");let r=this.wasm.allocStringBuffer(t.length);if(!r)throw this.wasm.free(n,t.length*4),new Error("Failed to allocate output buffer");try{new Uint32Array(this.memory.buffer,n,t.length).set(t);let s=this.wasm.readBoolAtIndices(e,n,t.length,r),i=new Uint8Array(s),o=new Uint8Array(this.memory.buffer,r,s);return i.set(o),i}finally{this.wasm.free(n,t.length*4),this.wasm.free(r,t.length)}}filterInt64(e,t,n){let r=Number(this.getRowCount(e));if(r===0)return new Uint32Array(0);let s=this.wasm.allocIndexBuffer(r);if(!s)throw new Error("Failed to allocate index buffer");try{let i=this.wasm.filterInt64Column(e,t,BigInt(n),s,r),o=new Uint32Array(i),c=new Uint32Array(this.memory.buffer,s,i);return o.set(c),o}finally{this.wasm.free(s,r*4)}}filterFloat64(e,t,n){let r=Number(this.getRowCount(e));if(r===0)return new Uint32Array(0);let s=this.wasm.allocIndexBuffer(r);if(!s)throw new Error("Failed to allocate index buffer");try{let i=this.wasm.filterFloat64Column(e,t,n,s,r),o=new Uint32Array(i),c=new Uint32Array(this.memory.buffer,s,i);return o.set(c),o}finally{this.wasm.free(s,r*4)}}readInt64AtIndices(e,t){if(t.length===0)return new BigInt64Array(0);let n=this.wasm.allocIndexBuffer(t.length);if(!n)throw new Error("Failed to allocate index buffer");let r=this.wasm.allocInt64Buffer(t.length);if(!r)throw this.wasm.free(n,t.length*4),new Error("Failed to allocate output buffer");try{new Uint32Array(this.memory.buffer,n,t.length).set(t);let s=this.wasm.readInt64AtIndices(e,n,t.length,r),i=new BigInt64Array(s),o=new BigInt64Array(this.memory.buffer,r,s);return i.set(o),i}finally{this.wasm.free(n,t.length*4),this.wasm.freeInt64Buffer(r,t.length)}}readFloat64AtIndices(e,t){if(t.length===0)return new Float64Array(0);let n=this.wasm.allocIndexBuffer(t.length);if(!n)throw new Error("Failed to allocate index buffer");let r=this.wasm.allocFloat64Buffer(t.length);if(!r)throw this.wasm.free(n,t.length*4),new Error("Failed to allocate output buffer");try{new Uint32Array(this.memory.buffer,n,t.length).set(t);let s=this.wasm.readFloat64AtIndices(e,n,t.length,r),i=new Float64Array(s),o=new Float64Array(this.memory.buffer,r,s);return i.set(o),i}finally{this.wasm.free(n,t.length*4),this.wasm.freeFloat64Buffer(r,t.length)}}sumInt64(e){return this.wasm.sumInt64Column(e)}sumFloat64(e){return this.wasm.sumFloat64Column(e)}minInt64(e){return this.wasm.minInt64Column(e)}maxInt64(e){return this.wasm.maxInt64Column(e)}avgFloat64(e){return this.wasm.avgFloat64Column(e)}debugStringColInfo(e){let t=this.wasm.debugStringColInfo(e);return{offsetsSize:Number(BigInt(t)>>32n),dataSize:Number(BigInt(t)&0xFFFFFFFFn)}}debugReadStringInfo(e,t){let n=this.wasm.debugReadStringInfo(e,t);if((n&0xFFFF0000n)===0xDEAD0000n){let r=Number(n&0xFFFFn);return{error:{1:"No file data",2:"No column entry",3:"Col meta out of bounds",4:"Not a string column",5:"Row out of bounds",6:"Invalid offset size"}[r]||`Unknown error ${r}`}}return{strStart:Number(BigInt(n)>>32n),strLen:Number(BigInt(n)&0xFFFFFFFFn)}}debugStringDataStart(e){let t=this.wasm.debugStringDataStart(e);return{dataStart:Number(BigInt(t)>>32n),fileLen:Number(BigInt(t)&0xFFFFFFFFn)}}getStringCount(e){return Number(this.wasm.getStringCount(e))}readStringAt(e,t){let r=this.wasm.allocStringBuffer(4096);if(!r)throw new Error("Failed to allocate string buffer");try{let s=this.wasm.readStringAt(e,t,r,4096);if(s===0)return"";let i=new Uint8Array(this.memory.buffer,r,Math.min(s,4096));return new TextDecoder().decode(i)}finally{this.wasm.free(r,4096)}}readStringColumn(e,t=1e3){let n=Math.min(this.getStringCount(e),t);if(n===0)return[];let r=[];for(let s=0;s<n;s++)r.push(this.readStringAt(e,s));return r}readStringsAtIndices(e,t){if(t.length===0)return[];let n=Math.min(t.length*256,256*1024),r=this.wasm.allocIndexBuffer(t.length);if(!r)throw new Error("Failed to allocate index buffer");let s=this.wasm.allocStringBuffer(n);if(!s)throw this.wasm.free(r,t.length*4),new Error("Failed to allocate string buffer");let i=this.wasm.allocU32Buffer(t.length);if(!i)throw this.wasm.free(r,t.length*4),this.wasm.free(s,n),new Error("Failed to allocate length buffer");try{new Uint32Array(this.memory.buffer,r,t.length).set(t);let o=this.wasm.readStringsAtIndices(e,r,t.length,s,n,i),c=new Uint32Array(this.memory.buffer,i,t.length),l=[],f=0;for(let u=0;u<t.length;u++){let h=c[u];if(h>0&&f+h<=o){let d=new Uint8Array(this.memory.buffer,s+f,h);l.push(new TextDecoder().decode(d)),f+=h}else l.push("")}return l}finally{this.wasm.free(r,t.length*4),this.wasm.free(s,n),this.wasm.free(i,t.length*4)}}getVectorInfo(e){let t=this.wasm.getVectorInfo(e);return{rows:Number(BigInt(t)>>32n),dimension:Number(BigInt(t)&0xFFFFFFFFn)}}readVectorAt(e,t){let n=this.getVectorInfo(e);if(n.dimension===0)return new Float32Array(0);let r=this.wasm.allocFloat32Buffer(n.dimension);if(!r)throw new Error("Failed to allocate vector buffer");try{let s=this.wasm.readVectorAt(e,t,r,n.dimension),i=new Float32Array(s),o=new Float32Array(this.memory.buffer,r,s);return i.set(o),i}finally{this.wasm.free(r,n.dimension*4)}}cosineSimilarity(e,t){if(e.length!==t.length)throw new Error("Vector dimensions must match");let n=this.wasm.allocFloat32Buffer(e.length),r=this.wasm.allocFloat32Buffer(t.length);if(!n||!r)throw new Error("Failed to allocate buffers");try{return new Float32Array(this.memory.buffer,n,e.length).set(e),new Float32Array(this.memory.buffer,r,t.length).set(t),this.wasm.cosineSimilarity(n,r,e.length)}finally{this.wasm.free(n,e.length*4),this.wasm.free(r,t.length*4)}}batchCosineSimilarity(e,t,n=!0){if(t.length===0)return new Float32Array(0);let r=e.length,s=t.length,i=this.wasm.allocFloat32Buffer(r),o=this.wasm.allocFloat32Buffer(s*r),c=this.wasm.allocFloat32Buffer(s);if(!i||!o||!c)throw new Error("Failed to allocate WASM buffers");try{new Float32Array(this.memory.buffer,i,r).set(e);let l=new Float32Array(this.memory.buffer,o,s*r);for(let u=0;u<s;u++)l.set(t[u],u*r);this.wasm.batchCosineSimilarity(i,o,r,s,c,n?1:0);let f=new Float32Array(s);return f.set(new Float32Array(this.memory.buffer,c,s)),f}finally{this.wasm.free(i,r*4),this.wasm.free(o,s*r*4),this.wasm.free(c,s*4)}}readAllVectors(e){let t=this.getVectorInfo(e);if(t.dimension===0||t.rows===0)return[];let n=t.dimension,r=t.rows,s=[],i=this.wasm.allocFloat32Buffer(r*n);if(!i)throw new Error("Failed to allocate vector buffer");try{if(this.wasm.readVectorColumn){let o=this.wasm.readVectorColumn(e,i,r*n),c=new Float32Array(this.memory.buffer,i,o);for(let l=0;l<r&&l*n<o;l++){let f=new Float32Array(n);f.set(c.subarray(l*n,(l+1)*n)),s.push(f)}}else for(let o=0;o<r;o++)s.push(this.readVectorAt(e,o));return s}finally{this.wasm.free(i,r*n*4)}}async vectorSearch(e,t,n=10,r=null){let s=t.length,o=this.getVectorInfo(e).rows,c=O();if(c.isAvailable()){r&&r(0,o);let u=this.readAllVectors(e);r&&r(o,o);let h=await c.batchCosineSimilarity(t,u,!0);return await Z().topK(h,null,n,!0)}r&&r(0,o);let l=this.readAllVectors(e);r&&r(o,o);let f=this.lanceql.batchCosineSimilarity(t,l,!0);return await Z().topK(f,null,n,!0)}df(){return new DataFrame(this)}};Qe(ee,"Op",{EQ:0,NE:1,LT:2,LE:3,GT:4,GE:5});var te=class{constructor(e="lanceql"){this.rootDir=e,this.root=null}async getRoot(){if(this.root)return this.root;if(typeof navigator>"u"||!navigator.storage?.getDirectory)throw new Error("OPFS not available. Requires modern browser with Origin Private File System support.");let e=await navigator.storage.getDirectory();return this.root=await e.getDirectoryHandle(this.rootDir,{create:!0}),this.root}async open(){return await this.getRoot(),this}async getDir(e){let t=await this.getRoot(),n=e.split("/").filter(s=>s),r=t;for(let s of n)r=await r.getDirectoryHandle(s,{create:!0});return r}async save(e,t){let n=e.split("/"),r=n.pop(),s=n.join("/"),o=await(s?await this.getDir(s):await this.getRoot()).getFileHandle(r,{create:!0});if(o.createSyncAccessHandle)try{let l=await o.createSyncAccessHandle();return l.truncate(0),l.write(t,{at:0}),l.flush(),l.close(),{path:e,size:t.byteLength}}catch{}let c=await o.createWritable();return await c.write(t),await c.close(),{path:e,size:t.byteLength}}async load(e){try{let t=e.split("/"),n=t.pop(),r=t.join("/"),c=await(await(await(r?await this.getDir(r):await this.getRoot()).getFileHandle(n)).getFile()).arrayBuffer();return new Uint8Array(c)}catch(t){if(t.name==="NotFoundError")return null;throw t}}async delete(e){try{let t=e.split("/"),n=t.pop(),r=t.join("/");return await(r?await this.getDir(r):await this.getRoot()).removeEntry(n),!0}catch(t){if(t.name==="NotFoundError")return!1;throw t}}async list(e=""){try{let t=e?await this.getDir(e):await this.getRoot(),n=[];for await(let[r,s]of t.entries())n.push({name:r,type:s.kind});return n}catch{return[]}}async exists(e){try{let t=e.split("/"),n=t.pop(),r=t.join("/");return await(r?await this.getDir(r):await this.getRoot()).getFileHandle(n),!0}catch{return!1}}async deleteDir(e){try{let t=e.split("/"),n=t.pop(),r=t.join("/");return await(r?await this.getDir(r):await this.getRoot()).removeEntry(n,{recursive:!0}),!0}catch{return!1}}async readRange(e,t,n){try{let r=e.split("/"),s=r.pop(),i=r.join("/"),u=await(await(await(i?await this.getDir(i):await this.getRoot()).getFileHandle(s)).getFile()).slice(t,t+n).arrayBuffer();return new Uint8Array(u)}catch(r){if(r.name==="NotFoundError")return null;throw r}}async getFileSize(e){try{let t=e.split("/"),n=t.pop(),r=t.join("/");return(await(await(r?await this.getDir(r):await this.getRoot()).getFileHandle(n)).getFile()).size}catch(t){if(t.name==="NotFoundError")return null;throw t}}async openFile(e){try{let t=e.split("/"),n=t.pop(),r=t.join("/"),i=await(r?await this.getDir(r):await this.getRoot()).getFileHandle(n);return new Oe(i)}catch(t){if(t.name==="NotFoundError")return null;throw t}}async isSupported(){try{return typeof navigator>"u"||!navigator.storage?.getDirectory?!1:(await navigator.storage.getDirectory(),!0)}catch{return!1}}async getStats(){try{let e=await this.getRoot(),t=0,n=0;async function r(s){for await(let[i,o]of s.entries())if(o.kind==="file"){let c=await o.getFile();t++,n+=c.size}else o.kind==="directory"&&await r(o)}return await r(e),{fileCount:t,totalSize:n}}catch{return{fileCount:0,totalSize:0}}}async listFiles(){try{let e=await this.getRoot(),t=[];async function n(r,s=""){for await(let[i,o]of r.entries())if(o.kind==="file"){let c=await o.getFile();t.push({name:s?`${s}/${i}`:i,size:c.size,lastModified:c.lastModified})}else o.kind==="directory"&&await n(o,s?`${s}/${i}`:i)}return await n(e),t}catch{return[]}}async clearAll(){try{let e=await this.getRoot(),t=0,n=[];for await(let[r,s]of e.entries())n.push({name:r,kind:s.kind});for(let r of n)await e.removeEntry(r.name,{recursive:r.kind==="directory"}),t++;return t}catch(e){return console.warn("Failed to clear OPFS:",e),0}}},Oe=class{constructor(e){this.fileHandle=e,this._file=null,this._size=null}async getFile(){return this._file||(this._file=await this.fileHandle.getFile(),this._size=this._file.size),this._file}async getSize(){return this._size===null&&await this.getFile(),this._size}async readRange(e,t){let s=await(await this.getFile()).slice(e,e+t).arrayBuffer();return new Uint8Array(s)}async readFromEnd(e){let t=await this.getSize();return this.readRange(t-e,e)}invalidate(){this._file=null,this._size=null}},Tn=new te;var ze=class{constructor(e=null,t={}){this.storage=e,this.cacheDir=t.cacheDir||"_cache",this.maxFileSize=t.maxFileSize||10*1024*1024,this.maxCacheSize=t.maxCacheSize||500*1024*1024,this.enabled=t.enabled??!0,this._stats={hits:0,misses:0,bytesFromCache:0,bytesFromNetwork:0},this._metaCache=new Map,this._metaCacheOrder=[],this.maxMetaCacheEntries=t.maxMetaCacheEntries||100}_setMetaCache(e,t){let n=this._metaCacheOrder.indexOf(e);for(n!==-1&&this._metaCacheOrder.splice(n,1);this._metaCacheOrder.length>=this.maxMetaCacheEntries;){let r=this._metaCacheOrder.shift();this._metaCache.delete(r)}this._metaCache.set(e,t),this._metaCacheOrder.push(e)}async init(){this.storage||(this.storage=new te,await this.storage.open())}_getCacheKey(e){let t=0;for(let n=0;n<e.length;n++){let r=e.charCodeAt(n);t=(t<<5)-t+r,t=t&t}return Math.abs(t).toString(36)}_getCachePath(e,t=""){let n=this._getCacheKey(e);return`${this.cacheDir}/${n}${t}`}async isCached(e){if(!this.enabled)return{cached:!1};try{await this.init();let t=this._getCachePath(e,"/meta.json"),n=await this.storage.load(t);return n?{cached:!0,meta:JSON.parse(new TextDecoder().decode(n))}:{cached:!1}}catch{return{cached:!1}}}async getFile(e,t=null){if(!this.enabled)return this._fetchFile(e);await this.init();let{cached:n,meta:r}=await this.isCached(e);if(n&&r.fullFile){let i=this._getCachePath(e,"/data.lance"),o=await this.storage.load(i);if(o)return this._stats.hits++,this._stats.bytesFromCache+=o.byteLength,console.log(`[HotTierCache] HIT: ${e} (${(o.byteLength/1024).toFixed(1)} KB)`),o}this._stats.misses++;let s=await this._fetchFile(e);return this._stats.bytesFromNetwork+=s.byteLength,s.byteLength<=this.maxFileSize&&await this._cacheFile(e,s),s}async getRange(e,t,n,r=null){if(!this.enabled)return this._fetchRange(e,t,n);let s=this._metaCache.get(e);if(s?.fullFileData){let o=s.fullFileData;if(o.byteLength>n)return this._stats.hits++,this._stats.bytesFromCache+=n-t+1,o.slice(t,n+1).buffer}if(await this.init(),!s){let{cached:o,meta:c}=await this.isCached(e);if(o&&c.fullFile){let l=this._getCachePath(e,"/data.lance"),f=await this.storage.load(l);if(f&&f.byteLength>n)return this._setMetaCache(e,{meta:c,fullFileData:f}),this._stats.hits++,this._stats.bytesFromCache+=n-t+1,f.slice(t,n+1).buffer}this._setMetaCache(e,{meta:o?c:null,fullFileData:null})}this._stats.misses++;let i=await this._fetchRange(e,t,n);return this._stats.bytesFromNetwork+=i.byteLength,i}async prefetch(e,t=null){await this.init();let{cached:n,meta:r}=await this.isCached(e);if(n&&r.fullFile){console.log(`[HotTierCache] Already cached: ${e}`);return}console.log(`[HotTierCache] Prefetching: ${e}`);let s=await this._fetchFile(e,t);await this._cacheFile(e,s),console.log(`[HotTierCache] Cached: ${e} (${(s.byteLength/1024/1024).toFixed(2)} MB)`)}async evict(e){await this.init();let t=this._getCachePath(e);await this.storage.delete(t),console.log(`[HotTierCache] Evicted: ${e}`)}async clear(){await this.init(),await this.storage.delete(this.cacheDir),this._stats={hits:0,misses:0,bytesFromCache:0,bytesFromNetwork:0},console.log("[HotTierCache] Cleared all cache")}getStats(){let e=this._stats.hits+this._stats.misses>0?(this._stats.hits/(this._stats.hits+this._stats.misses)*100).toFixed(1):0;return{...this._stats,hitRate:`${e}%`,bytesFromCacheMB:(this._stats.bytesFromCache/1024/1024).toFixed(2),bytesFromNetworkMB:(this._stats.bytesFromNetwork/1024/1024).toFixed(2)}}async _fetchFile(e,t=null){let n=await fetch(e);if(!n.ok)throw new Error(`HTTP error: ${n.status}`);if(t&&n.headers.get("content-length")){let s=parseInt(n.headers.get("content-length")),i=n.body.getReader(),o=[],c=0;for(;;){let{done:u,value:h}=await i.read();if(u)break;o.push(h),c+=h.length,t(c,s)}let l=new Uint8Array(c),f=0;for(let u of o)l.set(u,f),f+=u.length;return l}let r=await n.arrayBuffer();return new Uint8Array(r)}async _fetchRange(e,t,n){let r=await fetch(e,{headers:{Range:`bytes=${t}-${n}`}});if(!r.ok&&r.status!==206)throw new Error(`HTTP error: ${r.status}`);return r.arrayBuffer()}async _cacheFile(e,t){let n=this._getCachePath(e,"/meta.json"),r=this._getCachePath(e,"/data.lance"),s={url:e,size:t.byteLength,cachedAt:Date.now(),fullFile:!0,ranges:null};await this.storage.save(n,new TextEncoder().encode(JSON.stringify(s))),await this.storage.save(r,t)}async _cacheRange(e,t,n,r,s){let i=this._getCachePath(e,"/meta.json"),o=this._getCachePath(e,`/ranges/${t}-${n}`),c,{cached:l,meta:f}=await this.isCached(e);l?(c=f,c.ranges=c.ranges||[]):c={url:e,size:s,cachedAt:Date.now(),fullFile:!1,ranges:[]},c.ranges.push({start:t,end:n,cachedAt:Date.now()}),c.ranges=this._mergeRanges(c.ranges),await this.storage.save(i,new TextEncoder().encode(JSON.stringify(c))),await this.storage.save(o,r)}_mergeRanges(e){if(e.length<=1)return e;e.sort((n,r)=>n.start-r.start);let t=[e[0]];for(let n=1;n<e.length;n++){let r=t[t.length-1],s=e[n];s.start<=r.end+1?r.end=Math.max(r.end,s.end):t.push(s)}return t}},Le=null;function mt(){return Le||(Le=new ze),Le}async function gt(a){let e=[1,5,10,20,50,100],t=await Promise.all(e.map(async r=>{try{return(await fetch(`${a}/_versions/${r}.manifest`,{method:"HEAD"})).ok?r:0}catch{return 0}})),n=Math.max(...t);if(n===0)return null;for(let r=n+1;r<=n+30;r++)try{if((await fetch(`${a}/_versions/${r}.manifest`,{method:"HEAD"})).ok)n=r;else break}catch{break}return n}function pt(a){let t=new DataView(a.buffer,a.byteOffset).getUint32(0,!0),n=a.slice(4,4+t),r=0,s=null,i=null,o=(c,l)=>{let f=0,u=0,h=l;for(;h<c.length;){let d=c[h++];if(f|=(d&127)<<u,(d&128)===0)break;u+=7}return{value:f,pos:h}};for(;r<n.length;){let c=o(n,r);r=c.pos;let l=c.value>>3,f=c.value&7;if(f===2){let u=o(n,r);r=u.pos;let h=n.slice(r,r+u.value);if(r+=u.value,l===1){let d=Cn(h);d?.uuid&&(s=d.uuid,i=d.fieldId)}}else f===0?r=o(n,r).pos:f===5?r+=4:f===1&&(r+=8)}return s?{uuid:s,fieldId:i}:null}function Cn(a){let e=0,t=null,n=null,r=()=>{let s=0,i=0;for(;e<a.length;){let o=a[e++];if(s|=(o&127)<<i,(o&128)===0)break;i+=7}return s};for(;e<a.length;){let s=r(),i=s>>3,o=s&7;if(o===2){let c=r(),l=a.slice(e,e+c);e+=c,i===1&&(t=kn(l))}else if(o===0){let c=r();i===2&&(n=c)}else o===5?e+=4:o===1&&(e+=8)}return{uuid:t,fieldId:n}}function kn(a){let e=0;for(;e<a.length;){let t=a[e++],n=t>>3,r=t&7;if(r===2&&n===1){let s=a[e++],i=a.slice(e,e+s),o=Array.from(i).map(c=>c.toString(16).padStart(2,"0")).join("");return`${o.slice(0,8)}-${o.slice(8,12)}-${o.slice(12,16)}-${o.slice(16,20)}-${o.slice(20,32)}`}else if(r===0)for(;e<a.length&&a[e++]&128;);else r===5?e+=4:r===1&&(e+=8)}return null}function wt(a,e,t){let n=new t,r=yt(a);if(r&&(r.centroids&&(n.centroids=r.centroids.data,n.numPartitions=r.centroids.numPartitions,n.dimension=r.centroids.dimension),r.offsets?.length>0&&(n.partitionOffsets=r.offsets),r.lengths?.length>0&&(n.partitionLengths=r.lengths)),!n.centroids){let s=0,i=()=>{let o=0,c=0;for(;s<a.length;){let l=a[s++];if(o|=(l&127)<<c,(l&128)===0)break;c+=7}return o};for(;s<a.length-4;){let c=i()&7;if(c===2){let l=i();if(l>a.length-s)break;let f=a.slice(s,s+l);if(s+=l,l>100&&l<1e8){let u=bt(f);u&&(n.centroids=u.data,n.numPartitions=u.numPartitions,n.dimension=u.dimension)}}else c===0?i():c===5?s+=4:c===1&&(s+=8)}}return n.centroids?n:null}function yt(a){let e=0,t=[],n=[],r=null,s=()=>{let i=0,o=0;for(;e<a.length;){let c=a[e++];if(i|=(c&127)<<o,(c&128)===0)break;o+=7}return i};for(;e<a.length-4;){let i=e,o=s(),c=o>>3,l=o&7;if(l===2){let f=s();if(f>a.length-e||f<0){e=i+1;continue}let u=a.slice(e,e+f);if(e+=f,c===2&&f%8===0&&f>0){let h=new DataView(u.buffer,u.byteOffset,f);for(let d=0;d<f/8;d++)t.push(Number(h.getBigUint64(d*8,!0)))}else if(c===3)if(f%4===0&&f>0){let h=new DataView(u.buffer,u.byteOffset,f);for(let d=0;d<f/4;d++)n.push(h.getUint32(d*4,!0))}else{let h=0;for(;h<u.length;){let d=0,m=0;for(;h<u.length;){let g=u[h++];if(d|=(g&127)<<m,(g&128)===0)break;m+=7}n.push(d)}}else if(c===4)r=bt(u);else if(f>100){let h=yt(u);(h?.centroids||h?.offsets?.length>0)&&(h.centroids&&!r&&(r=h.centroids),h.offsets?.length>t.length&&(t=h.offsets),h.lengths?.length>n.length&&(n=h.lengths))}}else l===0?s():l===5?e+=4:l===1?e+=8:e=i+1}return r||t.length>0||n.length>0?{centroids:r,offsets:t,lengths:n}:null}function bt(a){let e=0,t=[],n=null,r=2,s=()=>{let i=0,o=0;for(;e<a.length;){let c=a[e++];if(i|=(c&127)<<o,(c&128)===0)break;o+=7}return i};for(;e<a.length;){let i=s(),o=i>>3,c=i&7;if(c===0){let l=s();o===1&&(r=l)}else if(c===2){let l=s(),f=a.slice(e,e+l);if(e+=l,o===2){let u=0;for(;u<f.length;){let h=0,d=0;for(;u<f.length;){let m=f[u++];if(h|=(m&127)<<d,(m&128)===0)break;d+=7}t.push(h)}}else o===3&&(n=f)}else c===5?e+=4:c===1&&(e+=8)}if(t.length>=2&&n&&r===2){let i=t[0],o=t[1];if(n.length===i*o*4)return{data:new Float32Array(n.buffer,n.byteOffset,i*o),numPartitions:i,dimension:o}}return null}async function Ge(a){let e;try{e=await fetch(a.auxiliaryUrl,{method:"HEAD"})}catch{return}if(!e.ok)return;let t=parseInt(e.headers.get("content-length"));if(!t)return;let n=await fetch(a.auxiliaryUrl,{headers:{Range:`bytes=${t-40}-${t-1}`}});if(!n.ok)return;let r=new Uint8Array(await n.arrayBuffer()),s=new DataView(r.buffer,r.byteOffset),i=Number(s.getBigUint64(0,!0)),o=Number(s.getBigUint64(8,!0)),c=Number(s.getBigUint64(16,!0)),l=s.getUint32(24,!0);if(new TextDecoder().decode(r.slice(36,40))!=="LANC")return;let u=l*16,h=await fetch(a.auxiliaryUrl,{headers:{Range:`bytes=${c}-${c+u-1}`}});if(!h.ok)return;let d=new Uint8Array(await h.arrayBuffer()),m=new DataView(d.buffer,d.byteOffset),g=[];for(let p=0;p<l;p++){let y=Number(m.getBigUint64(p*16,!0)),_=Number(m.getBigUint64(p*16+8,!0));g.push({offset:y,length:_})}if(g.length<2)return;a._auxBuffers=g,a._auxFileSize=t;let b=await fetch(a.auxiliaryUrl,{headers:{Range:`bytes=${o}-${c-1}`}});if(!b.ok)return;let w=new Uint8Array(await b.arrayBuffer());if(w.length>=32){let p=new DataView(w.buffer,w.byteOffset),y=Number(p.getBigUint64(0,!0)),_=Number(p.getBigUint64(8,!0)),x=await fetch(a.auxiliaryUrl,{headers:{Range:`bytes=${y}-${y+_-1}`}});if(x.ok){let v=new Uint8Array(await x.arrayBuffer());Ne(a,v)}}}function Ne(a,e){let t=0,n=[],r=()=>{let s=0,i=0;for(;t<e.length;){let o=e[t++];if(s|=(o&127)<<i,(o&128)===0)break;i+=7}return s};for(;t<e.length;){let s=r(),i=s>>3,o=s&7;if(o===2){let c=r();if(c>e.length-t)break;let l=e.slice(t,t+c);if(t+=c,i===2){let f=Mn(l);f&&n.push(f)}}else o===0?r():o===5?t+=4:o===1&&(t+=8)}a._columnPages=n}function Mn(a){let e=0,t=0,n=[],r=[],s=()=>{let i=0,o=0;for(;e<a.length;){let c=a[e++];if(i|=(c&127)<<o,(c&128)===0)break;o+=7}return i};for(;e<a.length;){let i=s(),o=i>>3,c=i&7;if(c===0){let l=s();o===3&&(t=l)}else if(c===2){let l=s(),f=a.slice(e,e+l);if(e+=l,o===1){let u=0;for(;u<f.length;){let h=0n,d=0n;for(;u<f.length;){let m=f[u++];if(h|=BigInt(m&127)<<d,(m&128)===0)break;d+=7n}n.push(Number(h))}}if(o===2){let u=0;for(;u<f.length;){let h=0n,d=0n;for(;u<f.length;){let m=f[u++];if(h|=BigInt(m&127)<<d,(m&128)===0)break;d+=7n}r.push(Number(h))}}}else c===5?e+=4:c===1&&(e+=8)}return{numRows:t,bufferOffsets:n,bufferSizes:r}}function _t(a,e){let t=0,n=()=>{let r=0,s=0;for(;t<e.length;){let i=e[t++];if(r|=(i&127)<<s,(i&128)===0)break;s+=7}return r};for(;t<e.length-4;){let r=n(),s=r>>3,i=r&7;if(i===2){let o=n();if(o>e.length-t)break;let c=e.slice(t,t+o);if(t+=o,s===2&&o>100&&o<2e3){let l=[],f=0;for(;f<c.length;){let u=0,h=0;for(;f<c.length;){let d=c[f++];if(u|=(d&127)<<h,(d&128)===0)break;h+=7}l.push(u)}l.length===a.numPartitions&&(a.partitionOffsets=l)}else if(s===3&&o>100&&o<2e3){let l=[],f=0;for(;f<c.length;){let u=0,h=0;for(;f<c.length;){let d=c[f++];if(u|=(d&127)<<h,(d&128)===0)break;h+=7}l.push(u)}l.length===a.numPartitions&&(a.partitionLengths=l)}}else if(i===0)n();else if(i===1)t+=8;else if(i===5)t+=4;else break}}var _e=class{constructor(e={}){this.maxSize=e.maxSize??50*1024*1024,this.currentSize=0,this.cache=new Map,this._head=null,this._tail=null}get(e){let t=this.cache.get(e);if(t)return this._moveToHead(t),t.data}delete(e){let t=this.cache.get(e);return t?(this._removeNode(t),this.cache.delete(e),this.currentSize-=t.size,!0):!1}set(e,t,n=null){return this.put(e,t,n)}put(e,t,n=null){let r=this.cache.get(e);r&&(this._removeNode(r),this.currentSize-=r.size,this.cache.delete(e));let s=n;for(s===null&&(t==null?s=0:t.byteLength!==void 0?s=t.byteLength:typeof t=="string"?s=t.length*2:typeof t=="object"?s=JSON.stringify(t).length*2:s=8);this.currentSize+s>this.maxSize&&this._tail;)this._evictTail();if(s>this.maxSize)return;let i={key:e,data:t,size:s,prev:null,next:null};this._addToHead(i),this.cache.set(e,i),this.currentSize+=s}_addToHead(e){e.prev=null,e.next=this._head,this._head&&(this._head.prev=e),this._head=e,this._tail||(this._tail=e)}_removeNode(e){e.prev?e.prev.next=e.next:this._head=e.next,e.next?e.next.prev=e.prev:this._tail=e.prev,e.prev=null,e.next=null}_moveToHead(e){e!==this._head&&(this._removeNode(e),this._addToHead(e))}_evictTail(){if(!this._tail)return;let e=this._tail;this._removeNode(e),this.cache.delete(e.key),this.currentSize-=e.size}clear(){this.cache.clear(),this._head=null,this._tail=null,this.currentSize=0}stats(){return{entries:this.cache.size,currentSize:this.currentSize,maxSize:this.maxSize,utilization:(this.currentSize/this.maxSize*100).toFixed(1)+"%"}}};var En=50*1024*1024;async function De(a){let e=`${a.datasetBaseUrl}/ivf_vectors.bin`;a.partitionVectorsUrl=e;let t=await fetch(e,{headers:{Range:"bytes=0-2055"}});if(!t.ok)return;let n=await t.arrayBuffer(),r=new BigUint64Array(n);a.partitionOffsets=Array.from(r,s=>Number(s)),a.hasPartitionIndex=!0}async function vt(a,e,t=384,n=null){if(!a.hasPartitionIndex||!a.partitionVectorsUrl)return null;let r=0,s=0,i=[],o=new Map;a._partitionCache||(a._partitionCache=new _e({maxSize:En}));for(let l of e){let f=a._partitionCache.get(l);f!==void 0?o.set(l,f):(i.push(l),r+=a.partitionOffsets[l+1]-a.partitionOffsets[l])}if(i.length===0)return xt(e,o,t,n);a._fetchStats||(a._fetchStats={concurrency:6,recentLatencies:[],minConcurrency:2,maxConcurrency:12});let c=a._fetchStats;for(let l=0;l<i.length;l+=c.concurrency){let f=i.slice(l,l+c.concurrency),u=performance.now(),h=await Promise.all(f.map(async g=>{let b=a.partitionOffsets[g],w=a.partitionOffsets[g+1],p=w-b;try{let y=await fetch(a.partitionVectorsUrl,{headers:{Range:`bytes=${b}-${w-1}`}});if(!y.ok)return{p:g,rowIds:[],vectors:[]};let _=await y.arrayBuffer(),v=new DataView(_).getUint32(0,!0),A=4+v*4,I=new Uint32Array(_.slice(4,A)),S=new Float32Array(_.slice(A));return s+=p,n&&n(s,r),{p:g,rowIds:Array.from(I),vectors:S,numVectors:v}}catch{return{p:g,rowIds:[],vectors:[]}}}));for(let g of h){let b={rowIds:g.rowIds,vectors:g.vectors,numVectors:g.numVectors??g.rowIds.length},w=g.rowIds.length*4+(g.vectors.byteLength||g.vectors.length*4);a._partitionCache.set(g.p,b,w),o.set(g.p,b)}let d=performance.now()-u;c.recentLatencies.push(d),c.recentLatencies.length>10&&c.recentLatencies.shift();let m=c.recentLatencies.reduce((g,b)=>g+b,0)/c.recentLatencies.length;m<50&&c.concurrency<c.maxConcurrency?c.concurrency++:m>200&&c.concurrency>c.minConcurrency&&c.concurrency--}return xt(e,o,t,n)}function xt(a,e,t,n){let r=0,s=0;for(let f of a){let u=e.get(f);u&&(r+=u.rowIds.length,s+=u.vectors.length)}let i=new Array(r),o=new Float32Array(s),c=0,l=0;for(let f of a){let u=e.get(f);if(u){for(let h=0;h<u.rowIds.length;h++)i[c++]=u.rowIds[h];o.set(u.vectors,l),l+=u.vectors.length}}return n&&n(100,100),{rowIds:i,vectors:o,preFlattened:!0}}async function $e(a){if(!a.auxiliaryUrl||!a._auxBufferOffsets||a._rowIdCacheReady)return;let e=a.partitionLengths.reduce((r,s)=>r+s,0);if(e===0)return;let t=a._auxBufferOffsets[1],n=e*8;try{let r=await fetch(a.auxiliaryUrl,{headers:{Range:`bytes=${t}-${t+n-1}`}});if(!r.ok)return;let s=new Uint8Array(await r.arrayBuffer()),i=new DataView(s.buffer,s.byteOffset);a._rowIdCache=new Map;let o=0;for(let c=0;c<a.partitionLengths.length;c++){let l=a.partitionLengths[c],f=[];for(let u=0;u<l;u++){let h=Number(i.getBigUint64(o*8,!0));f.push({fragId:Math.floor(h/4294967296),rowOffset:h%4294967296}),o++}a._rowIdCache.set(c,f)}a._rowIdCacheReady=!0}catch{}}async function At(a,e){if(a._rowIdCacheReady&&a._rowIdCache){let s=[];for(let i of e){let o=a._rowIdCache.get(i);if(o)for(let c of o)s.push({...c,partition:i})}return s}if(!a.auxiliaryUrl||!a._auxBufferOffsets)return null;let t=[];for(let s of e)s<a.partitionOffsets.length&&t.push({partition:s,startRow:a.partitionOffsets[s],numRows:a.partitionLengths[s]});if(t.length===0)return[];let n=[],r=a._auxBufferOffsets[1];for(let s of t){let i=r+s.startRow*8,o=i+s.numRows*8-1;try{let c=await fetch(a.auxiliaryUrl,{headers:{Range:`bytes=${i}-${o}`}});if(!c.ok)continue;let l=new Uint8Array(await c.arrayBuffer()),f=new DataView(l.buffer,l.byteOffset);for(let u=0;u<s.numRows;u++){let h=Number(f.getBigUint64(u*8,!0));n.push({fragId:Math.floor(h/4294967296),rowOffset:h%4294967296,partition:s.partition})}}catch{}}return n}function It(a,e){let t=0;for(let n of e)n<a.partitionLengths.length&&(t+=a.partitionLengths[n]);return t}var Ve=new Map,xe=new Map;function Ln(a,e){if(e>=a.length)return a;if(e<=0)return[];let t=0,n=a.length-1;for(;t<n;){let r=t+n>>1;a[r].score>a[t].score&&ne(a,t,r),a[n].score>a[t].score&&ne(a,t,n),a[r].score>a[n].score&&ne(a,r,n);let s=a[n].score,i=t;for(let o=t;o<n;o++)a[o].score>=s&&(ne(a,i,o),i++);if(ne(a,i,n),i===e-1)break;i<e-1?t=i+1:n=i-1}return a.slice(0,e)}function ne(a,e,t){let n=a[e];a[e]=a[t],a[t]=n}var j=class a{constructor(){this.centroids=null,this.numPartitions=0,this.dimension=0,this.partitionOffsets=[],this.partitionLengths=[],this.metricType="cosine",this.partitionIndexUrl=null,this.partitionStarts=null,this.hasPartitionIndex=!1,this._rowIdCache=null,this._rowIdCacheReady=!1,this._accessCounts=new Map}static async tryLoad(e){if(!e)return null;if(Ve.has(e))return Ve.get(e);if(xe.has(e))return xe.get(e);let t=a._doLoad(e);xe.set(e,t);try{let n=await t;return n&&Ve.set(e,n),n}finally{xe.delete(e)}}static async _doLoad(e){try{let t=await gt(e);if(!t)return null;let n=`${e}/_versions/${t}.manifest`,r=await fetch(n);if(!r.ok)return null;let s=await r.arrayBuffer(),i=pt(new Uint8Array(s));if(!i?.uuid)return null;let o=`${e}/_indices/${i.uuid}/index.idx`,c=await fetch(o);if(!c.ok)return null;let l=await c.arrayBuffer(),f=wt(new Uint8Array(l),i,a);if(!f)return null;f.auxiliaryUrl=`${e}/_indices/${i.uuid}/auxiliary.idx`,f.datasetBaseUrl=e;try{await Ge(f)}catch{}try{await De(f)}catch{}try{await $e(f)}catch{}return f}catch{return null}}async _loadPartitionIndex(){return De(this)}fetchPartitionData(e,t=384,n=null){return vt(this,e,t,n)}async _loadAuxiliaryMetadata(){return Ge(this)}_parseColumnMetaForPartitions(e){return Ne(this,e)}_parseAuxiliaryPartitionInfo(e){return _t(this,e)}async prefetchAllRowIds(){return $e(this)}fetchPartitionRowIds(e){return At(this,e)}getPartitionRowCount(e){return It(this,e)}findNearestPartitions(e,t=10){if(!this.centroids||e.length!==this.dimension)return[];t=Math.min(t,this.numPartitions);let n=new Array(this.numPartitions),r=0;for(let c=0;c<this.dimension;c++)r+=e[c]*e[c];let s=Math.sqrt(r);for(let c=0;c<this.numPartitions;c++){let l=c*this.dimension,f=0,u=0;for(let d=0;d<this.dimension;d++){let m=this.centroids[l+d];f+=e[d]*m,u+=m*m}let h=s*Math.sqrt(u);n[c]={idx:c,score:h===0?0:f/h}}let o=Ln(n,t).map(c=>c.idx);for(let c of o)this._accessCounts.set(c,(this._accessCounts.get(c)||0)+1);return o}async prefetchHotPartitions(e=10,t=3){if(!this._accessCounts||this._accessCounts.size===0)return;let n=[...this._accessCounts.entries()].filter(([r,s])=>s>=t).sort((r,s)=>s[1]-r[1]).slice(0,e).map(([r])=>r);n.length!==0&&await this.fetchPartitionData(n,this.dimension)}getAccessStats(){return{totalPartitions:this.numPartitions,accessedPartitions:this._accessCounts.size,topPartitions:[...this._accessCounts.entries()].sort((e,t)=>t[1]-e[1]).slice(0,10)}}};async function St(a){let e=a.url.match(/^(.+\.lance)\/data\/.+\.lance$/);if(e){a._datasetBaseUrl=e[1];try{let t=`${a._datasetBaseUrl}/_versions/1.manifest`,n=await fetch(t);if(!n.ok)return;let r=await n.arrayBuffer();a._schema=zn(new Uint8Array(r))}catch{}}}function zn(a){let e=new DataView(a.buffer,a.byteOffset),t=e.getUint32(0,!0),n=4+t,r;if(n+4<a.length){let c=e.getUint32(n,!0);c>0&&n+4+c<=a.length?r=a.slice(n+4,n+4+c):r=a.slice(4,4+t)}else r=a.slice(4,4+t);let s=0,i=[],o=()=>{let c=0,l=0;for(;s<r.length;){let f=r[s++];if(c|=(f&127)<<l,(f&128)===0)break;l+=7}return c};for(;s<r.length;){let c=o(),l=c>>3,f=c&7;if(l===1&&f===2){let u=o(),h=s+u,d=null,m=null,g=null;for(;s<h;){let b=o(),w=b>>3,p=b&7;if(p===0){let y=o();w===3&&(m=y)}else if(p===2){let y=o(),_=r.slice(s,s+y);s+=y,w===2?d=new TextDecoder().decode(_):w===5&&(g=new TextDecoder().decode(_))}else p===5?s+=4:p===1&&(s+=8)}d&&i.push({name:d,id:m,type:g})}else if(f===0)o();else if(f===2){let u=o();s+=u}else f===5?s+=4:f===1&&(s+=8)}return i}function Ft(a){return a._schema&&a._schema.length>0?a._schema.map(e=>e.name):Array.from({length:a._numColumns},(e,t)=>`column_${t}`)}async function Ut(a){if(a._columnTypes)return a._columnTypes;let e=[];if(a._schema&&a._schema.length>0){for(let t=0;t<a._numColumns;t++){let n=a._schema[t],r=n?.type?.toLowerCase()||"",s=n?.name?.toLowerCase()||"",i="unknown",o=s.includes("embedding")||s.includes("vector")||s.includes("emb")||s==="vec";r.includes("utf8")||r.includes("string")||r.includes("large_utf8")?i="string":r.includes("fixed_size_list")||r.includes("vector")||o?i="vector":r.includes("int64")||r==="int64"?i="int64":r.includes("int32")||r==="int32"?i="int32":r.includes("int16")||r==="int16"?i="int16":r.includes("int8")||r==="int8"?i="int8":r.includes("float64")||r.includes("double")?i="float64":r.includes("float32")||r.includes("float")&&!r.includes("64")?i="float32":r.includes("bool")&&(i="bool"),e.push(i)}if(e.some(t=>t!=="unknown"))return a._columnTypes=e,e;e.length=0}for(let t=0;t<a._numColumns;t++){let n="unknown",r=a.columnNames[t]?.toLowerCase()||"",s=r.includes("embedding")||r.includes("vector")||r.includes("emb")||r==="vec";try{await a.readStringAt(t,0),n="string",e.push(n);continue}catch{}try{let i=await a.getColumnOffsetEntry(t);if(i.len>0){let o=await a.fetchRange(i.pos,i.pos+i.len-1),c=new Uint8Array(o),l=a._parseColumnMeta(c);if(l.rows>0&&l.size>0){let f=l.size/l.rows;if(s&&f>=4)n="vector";else if(f===8)n="int64";else if(f===4)try{let u=await a.readInt32AtIndices(t,[0]);if(u.length>0){let h=u[0];h>=-1e6&&h<=1e6&&Number.isInteger(h)?n="int32":n="float32"}}catch{n="float32"}else f>8&&f%4===0?n="vector":f===2?n="int16":f===1&&(n="int8")}}}catch{}e.push(n)}return a._columnTypes=e,e}function Pt(a){let e=0,t=[],n=0,r=()=>{let u=0n,h=0n;for(;e<a.length;){let d=a[e++];if(u|=BigInt(d&127)<<h,(d&128)===0)break;h+=7n}return Number(u)};for(;e<a.length;){let u=r(),h=u>>3,d=u&7;if(h===2&&d===2){let m=r(),g=e+m,b=[],w=[],p=0;for(;e<g;){let y=r(),_=y>>3,x=y&7;if(_===1&&x===2){let v=r(),A=e+v;for(;e<A;)b.push(r())}else if(_===2&&x===2){let v=r(),A=e+v;for(;e<A;)w.push(r())}else if(_===3&&x===0)p=r();else if(x===0)r();else if(x===2){let v=r();e+=v}else x===5?e+=4:x===1&&(e+=8)}t.push({offsets:b,sizes:w,rows:p}),n+=p}else if(d===0)r();else if(d===2){let m=r();e+=m}else d===5?e+=4:d===1&&(e+=8)}let s=t[0]||{offsets:[],sizes:[],rows:0},i=s.offsets,o=s.sizes,c=0;for(let u of t){let h=u.sizes.length>1?1:0;c+=u.sizes[h]||0}let l=i.length>1?1:0,f=i.length>1?0:-1;return{offset:i[l]||0,size:t.length>1?c:o[l]||0,rows:n,nullBitmapOffset:f>=0?i[f]:null,nullBitmapSize:f>=0?o[f]:null,bufferOffsets:i,bufferSizes:o,pages:t}}function re(a){let e=[],t=0,n=()=>{let s=0,i=0;for(;t<a.length;){let o=a[t++];if(s|=(o&127)<<i,(o&128)===0)break;i+=7}return s};for(;t<a.length;){let s=n(),i=s>>3,o=s&7;if(i===2&&o===2){let c=n(),l=t+c,f=[0,0],u=[0,0],h=0;for(;t<l;){let d=n(),m=d>>3,g=d&7;if(m===1&&g===2){let b=n(),w=t+b,p=0;for(;t<w&&p<2;)f[p++]=n();t=w}else if(m===2&&g===2){let b=n(),w=t+b,p=0;for(;t<w&&p<2;)u[p++]=n();t=w}else if(m===3&&g===0)h=n();else if(m===4&&g===2){let b=n();t+=b}else if(g===0)n();else if(g===2){let b=n();t+=b}else g===5?t+=4:g===1&&(t+=8)}e.push({offsetsStart:f[0],offsetsSize:u[0],dataStart:f[1],dataSize:u[1],rows:h})}else if(o===0)n();else if(o===2){let c=n();t+=c}else o===5?t+=4:o===1&&(t+=8)}return{...e[0]||{offsetsStart:0,offsetsSize:0,dataStart:0,dataSize:0,rows:0},pages:e}}function E(a,e,t=1024){if(a.length===0)return[];let n=[...a].map((i,o)=>({idx:i,origPos:o}));n.sort((i,o)=>i.idx-o.idx);let r=[],s=0;for(let i=1;i<=n.length;i++)(i===n.length||(n[i].idx-n[i-1].idx)*e>t)&&(r.push({startIdx:n[s].idx,endIdx:n[i-1].idx,items:n.slice(s,i)}),s=i);return r}async function Tt(a,e,t){if(t.length===0)return new BigInt64Array(0);let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r)),i=new BigInt64Array(t.length),o=8,c=E(t,o);return await Promise.all(c.map(async l=>{let f=s.offset+l.startIdx*o,u=s.offset+(l.endIdx+1)*o-1,h=await a.fetchRange(f,u),d=new DataView(h);for(let m of l.items){let g=(m.idx-l.startIdx)*o;i[m.origPos]=d.getBigInt64(g,!0)}})),i}async function Ct(a,e,t){if(t.length===0)return new Float64Array(0);let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r)),i=new Float64Array(t.length),o=8,c=E(t,o);return await Promise.all(c.map(async l=>{let f=s.offset+l.startIdx*o,u=s.offset+(l.endIdx+1)*o-1,h=await a.fetchRange(f,u),d=new DataView(h);for(let m of l.items){let g=(m.idx-l.startIdx)*o;i[m.origPos]=d.getFloat64(g,!0)}})),i}async function kt(a,e,t){if(t.length===0)return new Int32Array(0);let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r)),i=new Int32Array(t.length),o=4,c=E(t,o);return await Promise.all(c.map(async l=>{let f=s.offset+l.startIdx*o,u=s.offset+(l.endIdx+1)*o-1,h=await a.fetchRange(f,u),d=new DataView(h);for(let m of l.items){let g=(m.idx-l.startIdx)*o;i[m.origPos]=d.getInt32(g,!0)}})),i}async function Bt(a,e,t){if(t.length===0)return new Float32Array(0);let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r)),i=new Float32Array(t.length),o=4,c=E(t,o);return await Promise.all(c.map(async l=>{let f=s.offset+l.startIdx*o,u=s.offset+(l.endIdx+1)*o-1,h=await a.fetchRange(f,u),d=new DataView(h);for(let m of l.items){let g=(m.idx-l.startIdx)*o;i[m.origPos]=d.getFloat32(g,!0)}})),i}async function Mt(a,e,t){if(t.length===0)return new Int16Array(0);let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r)),i=new Int16Array(t.length),o=2,c=E(t,o);return await Promise.all(c.map(async l=>{let f=s.offset+l.startIdx*o,u=s.offset+(l.endIdx+1)*o-1,h=await a.fetchRange(f,u),d=new DataView(h);for(let m of l.items){let g=(m.idx-l.startIdx)*o;i[m.origPos]=d.getInt16(g,!0)}})),i}async function Rt(a,e,t){if(t.length===0)return new Uint8Array(0);let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r)),i=new Uint8Array(t.length),o=1,c=E(t,o);return await Promise.all(c.map(async l=>{let f=s.offset+l.startIdx*o,u=s.offset+(l.endIdx+1)*o-1,h=await a.fetchRange(f,u),d=new Uint8Array(h);for(let m of l.items){let g=m.idx-l.startIdx;i[m.origPos]=d[g]}})),i}async function Et(a,e,t){if(t.length===0)return new Uint8Array(0);let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r)),i=new Uint8Array(t.length),o=t.map(g=>Math.floor(g/8)),c=[...new Set(o)].sort((g,b)=>g-b);if(c.length===0)return i;let l=c[0],f=c[c.length-1],u=s.offset+l,h=s.offset+f,d=await a.fetchRange(u,h),m=new Uint8Array(d);for(let g=0;g<t.length;g++){let b=t[g],w=Math.floor(b/8),p=b%8,y=w-l;y>=0&&y<m.length&&(i[g]=m[y]>>p&1)}return i}async function Ot(a,e,t){let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=re(new Uint8Array(r));if(s.offsetsSize===0||s.dataSize===0)throw new Error(`Not a string column - offsetsSize=${s.offsetsSize}, dataSize=${s.dataSize}`);let i=s.offsetsSize/s.rows;if(i!==4&&i!==8)throw new Error(`Not a string column - bytesPerOffset=${i}, expected 4 or 8`);if(t>=s.rows)return"";let o=i,c=s.offsetsStart+t*o,l=await a.fetchRange(c,c+o*2-1),f=new DataView(l),u,h;if(o===4?(u=f.getUint32(0,!0),h=f.getUint32(4,!0)):(u=Number(f.getBigUint64(0,!0)),h=Number(f.getBigUint64(8,!0))),h<=u)return"";let d=h-u,m=await a.fetchRange(s.dataStart+u,s.dataStart+h-1);return new TextDecoder().decode(m)}async function Lt(a,e,t){if(t.length===0)return[];let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=re(new Uint8Array(r));if(!s.pages||s.pages.length===0)return t.map(()=>"");let i=new Array(t.length).fill(""),o=0,c=[];for(let f of s.pages){if(f.offsetsSize===0||f.dataSize===0||f.rows===0){o+=f.rows;continue}c.push({start:o,end:o+f.rows,page:f}),o+=f.rows}let l=new Map;for(let f=0;f<t.length;f++){let u=t[f];for(let h=0;h<c.length;h++){let d=c[h];if(u>=d.start&&u<d.end){l.has(h)||l.set(h,[]),l.get(h).push({globalIdx:u,localIdx:u-d.start,resultIdx:f});break}}}for(let[f,u]of l){let d=c[f].page,m=d.offsetsSize/d.rows;if(m!==4&&m!==8)continue;u.sort((p,y)=>p.localIdx-y.localIdx);let g=[],b=0;for(let p=1;p<=u.length;p++)(p===u.length||u[p].localIdx-u[p-1].localIdx>100)&&(g.push(u.slice(b,p)),b=p);let w=[];if(await Promise.all(g.map(async p=>{let y=p[0].localIdx,_=p[p.length-1].localIdx,x=y>0?y-1:0,v=_,A=d.offsetsStart+x*m,I=d.offsetsStart+(v+1)*m-1,S=await a.fetchRange(A,I),U=new DataView(S);for(let L of p){let B=L.localIdx-x,C,k;m===4?(k=U.getUint32(B*4,!0),C=L.localIdx===0?0:U.getUint32((B-1)*4,!0)):(k=Number(U.getBigUint64(B*8,!0)),C=L.localIdx===0?0:Number(U.getBigUint64((B-1)*8,!0))),k>C&&w.push({start:C,end:k,resultIdx:L.resultIdx,dataStart:d.dataStart})}})),w.length>0){w.sort((_,x)=>_.start-x.start);let p=[],y=0;for(let _=1;_<=w.length;_++)(_===w.length||w[_].start-w[_-1].end>4096)&&(p.push({rangeStart:w[y].start,rangeEnd:w[_-1].end,items:w.slice(y,_),dataStart:w[y].dataStart}),y=_);await Promise.all(p.map(async _=>{let x=await a.fetchRange(_.dataStart+_.rangeStart,_.dataStart+_.rangeEnd-1),v=new Uint8Array(x);for(let A of _.items){let I=A.start-_.rangeStart,S=A.end-A.start,U=v.slice(I,I+S);i[A.resultIdx]=new TextDecoder().decode(U)}}))}}return i}async function qe(a,e){let t=await a.getColumnOffsetEntry(e);if(t.len===0)return{rows:0,dimension:0};let n=await a.fetchRange(t.pos,t.pos+t.len-1),r=a._parseColumnMeta(new Uint8Array(n));if(r.rows===0)return{rows:0,dimension:0};let s=0;if(r.pages&&r.pages.length>0){let i=r.pages[0],o=i.sizes.length>1?1:0,c=i.sizes[o]||0,l=i.rows||0;l>0&&c>0&&(s=Math.floor(c/(l*4)))}else r.size>0&&(s=Math.floor(r.size/(r.rows*4)));return{rows:r.rows,dimension:s}}async function zt(a,e,t){let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r));if(s.rows===0)return new Float32Array(0);if(t>=s.rows)return new Float32Array(0);let i=Math.floor(s.size/(s.rows*4));if(i===0)return new Float32Array(0);let o=s.offset+t*i*4,c=o+i*4-1,l=await a.fetchRange(o,c);return new Float32Array(l)}async function He(a,e,t){if(t.length===0)return[];let n=await a.getColumnOffsetEntry(e),r=await a.fetchRange(n.pos,n.pos+n.len-1),s=a._parseColumnMeta(new Uint8Array(r));if(s.rows===0)return t.map(()=>new Float32Array(0));let i=Math.floor(s.size/(s.rows*4));if(i===0)return t.map(()=>new Float32Array(0));let o=i*4,c=new Array(t.length),l=E(t,o,o*50),f=6;for(let u=0;u<l.length;u+=f){let h=l.slice(u,u+f);await Promise.all(h.map(async d=>{try{let m=s.offset+d.startIdx*o,g=s.offset+(d.endIdx+1)*o-1,b=await a.fetchRange(m,g);for(let w of d.items){let p=(w.idx-d.startIdx)*o;c[w.origPos]=new Float32Array(b.slice(p,p+o))}}catch{for(let g of d.items)c[g.origPos]=new Float32Array(0)}}))}return c}function Gt(a,e){if(a.length!==e.length)return 0;let t=0,n=0,r=0;for(let i=0;i<a.length;i++)t+=a[i]*e[i],n+=a[i]*a[i],r+=e[i]*e[i];let s=Math.sqrt(n)*Math.sqrt(r);return s===0?0:t/s}async function Nt(a,e,t,n=10,r=null,s={}){let{nprobe:i=10}=s,o=await qe(a,e);if(o.dimension===0||o.dimension!==t.length)throw new Error(`Dimension mismatch: query=${t.length}, column=${o.dimension}`);if(!a.hasIndex())throw new Error("No IVF index found. Vector search requires an IVF index for efficient querying.");if(a._ivfIndex.dimension!==t.length)throw new Error(`Query dimension (${t.length}) does not match index dimension (${a._ivfIndex.dimension}).`);return await Dn(a,e,t,n,i,r)}async function Dn(a,e,t,n,r,s){s&&s(0,100);let i=a._ivfIndex.findNearestPartitions(t,r),o=await a._ivfIndex.fetchPartitionRowIds(i);if(o&&o.length>0)return await $n(a,e,t,n,o,s);throw new Error("Failed to fetch row IDs from IVF index. Dataset may be missing auxiliary.idx or ivf_partitions.bin.")}async function $n(a,e,t,n,r,s){let i=t.length,o=new Map;for(let g of r)o.has(g.fragId)||o.set(g.fragId,[]),o.get(g.fragId).push(g.rowOffset);let c=[],l=[],f=0,u=r.length;for(let[g,b]of o){s&&s(f,u);let w=await He(a,e,b);for(let p=0;p<b.length;p++){let y=w[p];y&&y.length===i&&(c.push(y),l.push(g*5e4+b[p])),f++}}let h,d=O();d.isAvailable()&&(h=await d.batchCosineSimilarity(t,c,!0)),h||(h=a.lanceql.batchCosineSimilarity(t,c,!0));let m=[];for(let g=0;g<h.length;g++){let b=h[g],w=l[g];m.length<n?(m.push({idx:w,score:b}),m.sort((p,y)=>y.score-p.score)):b>m[n-1].score&&(m[n-1]={idx:w,score:b},m.sort((p,y)=>y.score-p.score))}return s&&s(u,u),{indices:m.map(g=>g.idx),scores:m.map(g=>g.score),usedIndex:!0,searchedRows:c.length}}async function Dt(a,e){let t=await a.getColumnOffsetEntry(e),n=await a.fetchRange(t.pos,t.pos+t.len-1),r=a._parseColumnMeta(new Uint8Array(n));if(!r.pages||r.pages.length===0||r.rows===0)return new Float32Array(0);let s=r.pages[0],i=s.sizes.length>1?1:0,o=s.sizes[i]||0,c=s.rows||0;if(c===0||o===0)return new Float32Array(0);let l=Math.floor(o/(c*4));if(l===0)return new Float32Array(0);let f=r.rows,u=new Float32Array(f*l),h=r.pages.map(async(g,b)=>{let w=g.sizes.length>1?1:0,p=g.offsets[w]||0,y=g.sizes[w]||0;if(y===0)return{pageIdx:b,data:new Float32Array(0),rows:0};let _=await a.fetchRange(p,p+y-1),x=new Float32Array(_);return{pageIdx:b,data:x,rows:g.rows}}),d=await Promise.all(h),m=0;for(let g of d.sort((b,w)=>b.pageIdx-w.pageIdx))u.set(g.data,m),m+=g.rows*l;return u}async function $t(a,{offset:e=0,limit:t=50,columns:n=null}={}){let r=n||Array.from({length:a._numColumns},(h,d)=>d),s=await a.getRowCount(0),i=Math.min(e,s),o=Math.min(t,s-i);if(o<=0)return{columns:r.map(()=>[]),columnNames:a.columnNames.slice(0,r.length),total:s};let c=Array.from({length:o},(h,d)=>i+d),l=await a.detectColumnTypes(),f=r.map(async h=>{let d=l[h]||"unknown";try{switch(d){case"string":case"utf8":case"large_utf8":return await a.readStringsAtIndices(h,c);case"int64":return Array.from(await a.readInt64AtIndices(h,c));case"int32":return Array.from(await a.readInt32AtIndices(h,c));case"int16":return Array.from(await a.readInt16AtIndices(h,c));case"uint8":return Array.from(await a.readUint8AtIndices(h,c));case"float64":case"double":return Array.from(await a.readFloat64AtIndices(h,c));case"float32":case"float":return Array.from(await a.readFloat32AtIndices(h,c));case"bool":case"boolean":return await a.readBoolAtIndices(h,c);case"fixed_size_list":case"vector":let m=await a.readVectorsAtIndices(h,c);return Array.isArray(m)?m:Array.from(m);default:return await a.readStringsAtIndices(h,c)}}catch{return c.map(()=>null)}});return{columns:await Promise.all(f),columnNames:r.map(h=>a.columnNames[h]||`column_${h}`),total:s}}var W=class a{constructor(e,t,n,r){this.lanceql=e,this.wasm=e.wasm,this.memory=e.memory,this.url=t,this.fileSize=n;let s=new Uint8Array(r);if(this.footerPtr=this.wasm.alloc(s.length),!this.footerPtr)throw new Error("Failed to allocate memory for footer");this.footerLen=s.length,new Uint8Array(this.memory.buffer).set(s,this.footerPtr),this._numColumns=this.wasm.parseFooterGetColumns(this.footerPtr,this.footerLen),this._majorVersion=this.wasm.parseFooterGetMajorVersion(this.footerPtr,this.footerLen),this._minorVersion=this.wasm.parseFooterGetMinorVersion(this.footerPtr,this.footerLen),this._columnMetaStart=this.wasm.getColumnMetaStart(this.footerPtr,this.footerLen),this._columnMetaOffsetsStart=this.wasm.getColumnMetaOffsetsStart(this.footerPtr,this.footerLen),this._columnMetaCache=new Map,this._columnOffsetCache=new Map,this._columnTypes=null,this._schema=null,this._datasetBaseUrl=null,this._ivfIndex=null}static async open(e,t){let n=await fetch(t,{method:"HEAD"});if(!n.ok)throw new Error(`HTTP error: ${n.status}`);let r=n.headers.get("Content-Length");if(!r)throw new Error("Server did not return Content-Length");let s=parseInt(r,10),o=s-40,c=await fetch(t,{headers:{Range:`bytes=${o}-${s-1}`}});if(!c.ok&&c.status!==206)throw new Error(`HTTP error: ${c.status}`);let l=await c.arrayBuffer(),f=new Uint8Array(l),u=String.fromCharCode(f[36],f[37],f[38],f[39]);if(u!=="LANC")throw new Error(`Invalid Lance file: expected LANC magic, got "${u}"`);let h=new a(e,t,s,l);await St(h);let d=t.includes("/data/");return d||await h._tryLoadIndex(),d||console.log(`[LanceQL] Loaded: ${h._numColumns} columns, ${(s/1024/1024).toFixed(1)}MB, schema: ${h._schema?"yes":"no"}, index: ${h.hasIndex()?"yes":"no"}`),h}async _tryLoadIndex(){if(this._datasetBaseUrl)try{this._ivfIndex=await j.tryLoad(this._datasetBaseUrl)}catch{}}hasIndex(){return this._ivfIndex!==null&&this._ivfIndex.centroids!==null}get columnNames(){return Ft(this)}get schema(){return this._schema}get datasetBaseUrl(){return this._datasetBaseUrl}get numColumns(){return this._numColumns}get size(){return this.fileSize}get version(){return{major:this._majorVersion,minor:this._minorVersion}}get columnMetaStart(){return Number(this._columnMetaStart)}get columnMetaOffsetsStart(){return Number(this._columnMetaOffsetsStart)}async fetchRange(e,t){(e<0||t<e||t>=this.size)&&console.error(`Invalid range: ${e}-${t}, file size: ${this.size}`);let n=mt();if(n.enabled){let i=await n.getRange(this.url,e,t,this.size);return this._onFetch&&this._onFetch(i.byteLength,1),i}let r=await fetch(this.url,{headers:{Range:`bytes=${e}-${t}`}});if(!r.ok&&r.status!==206)throw console.error(`Fetch failed: ${r.status} for range ${e}-${t}`),new Error(`HTTP error: ${r.status}`);let s=await r.arrayBuffer();return this._onFetch&&this._onFetch(s.byteLength,1),s}onFetch(e){this._onFetch=e}close(){this.footerPtr&&(this.wasm.free(this.footerPtr,this.footerLen),this.footerPtr=null)}async getColumnOffsetEntry(e){if(e>=this._numColumns)return{pos:0,len:0};if(this._columnOffsetCache.has(e))return this._columnOffsetCache.get(e);let t=this.columnMetaOffsetsStart+e*16,n=await this.fetchRange(t,t+15),r=new DataView(n),s={pos:Number(r.getBigUint64(0,!0)),len:Number(r.getBigUint64(8,!0))};return this._columnOffsetCache.set(e,s),s}async getColumnDebugInfo(e){let t=await this.getColumnOffsetEntry(e);if(t.len===0)return{offset:0,size:0,rows:0};let n=await this.fetchRange(t.pos,t.pos+t.len-1),r=new Uint8Array(n);return this._parseColumnMeta(r)}_parseColumnMeta(e){return Pt(e)}_parseStringColumnMeta(e){return re(e)}_batchIndices(e,t,n=1024){return E(e,t,n)}async _getCachedColumnMeta(e){if(this._columnMetaCache.has(e))return this._columnMetaCache.get(e);let t=await this.getColumnOffsetEntry(e);if(t.len===0)return null;let n=await this.fetchRange(t.pos,t.pos+t.len-1),r=new Uint8Array(n);return this._columnMetaCache.set(e,r),r}readInt64AtIndices(e,t){return Tt(this,e,t)}readFloat64AtIndices(e,t){return Ct(this,e,t)}readInt32AtIndices(e,t){return kt(this,e,t)}readFloat32AtIndices(e,t){return Bt(this,e,t)}readInt16AtIndices(e,t){return Mt(this,e,t)}readUint8AtIndices(e,t){return Rt(this,e,t)}readBoolAtIndices(e,t){return Et(this,e,t)}readStringAt(e,t){return Ot(this,e,t)}readStringsAtIndices(e,t){return Lt(this,e,t)}async getRowCount(e){return(await this.getColumnDebugInfo(e)).rows}detectColumnTypes(){return Ut(this)}getVectorInfo(e){return qe(this,e)}readVectorAt(e,t){return zt(this,e,t)}readVectorsAtIndices(e,t){return He(this,e,t)}cosineSimilarity(e,t){return Gt(e,t)}vectorSearch(e,t,n=10,r=null,s={}){return Nt(this,e,t,n,r,s)}readVectorColumn(e){return Dt(this,e)}readRows(e={}){return $t(this,e)}};var se=class{constructor(e="lanceql-cache",t=1){this.dbName=e,this.version=t,this.db=null}async open(){return this.db?this.db:new Promise((e,t)=>{let n=indexedDB.open(this.dbName,this.version);n.onerror=()=>t(n.error),n.onsuccess=()=>{this.db=n.result,e(this.db)},n.onupgradeneeded=r=>{let s=r.target.result;s.objectStoreNames.contains("datasets")||s.createObjectStore("datasets",{keyPath:"url"}).createIndex("timestamp","timestamp")}})}async get(e){try{let t=await this.open();return new Promise(n=>{let i=t.transaction("datasets","readonly").objectStore("datasets").get(e);i.onsuccess=()=>n(i.result||null),i.onerror=()=>n(null)})}catch(t){return console.warn("[MetadataCache] Get failed:",t),null}}async set(e,t){try{let n=await this.open();return new Promise((r,s)=>{let o=n.transaction("datasets","readwrite").objectStore("datasets"),c={url:e,timestamp:Date.now(),...t},l=o.put(c);l.onsuccess=()=>r(),l.onerror=()=>s(l.error)})}catch(n){console.warn("[MetadataCache] Set failed:",n)}}async delete(e){try{let t=await this.open();return new Promise(n=>{let r=t.transaction("datasets","readwrite");r.objectStore("datasets").delete(e),r.oncomplete=()=>n()})}catch(t){console.warn("[MetadataCache] Delete failed:",t)}}async clear(){try{let e=await this.open();return new Promise(t=>{let n=e.transaction("datasets","readwrite");n.objectStore("datasets").clear(),n.oncomplete=()=>t()})}catch(e){console.warn("[MetadataCache] Clear failed:",e)}}},Bs=new se;function Vt(a,e,t){let n=0,r=0,s=0,i=0,o=0,c=()=>{let u=0,h=0;for(;o<a.length;){let d=a[o++];if(u|=(d&127)<<h,(d&128)===0)break;h+=7}return u};for(;o<a.length;){let u=c(),h=u>>3,d=u&7;if(d===0){let m=c();h===1?n=m:h===2?r=m:h===3?s=m:h===4&&(i=m)}else if(d===2){let m=c();o+=m}else d===5?o+=4:d===1&&(o+=8)}if(i===0)return null;let f=`_deletions/${e}-${r}-${s}.${n===0?"arrow":"bin"}`;return{fileType:n===0?"arrow":"bitmap",readVersion:r,id:s,numDeletedRows:i,path:f,url:`${t}/${f}`}}function qn(a){let e=new Set,t=0;a.length>=8&&String.fromCharCode(...a.slice(0,6))==="ARROW1"&&(t=8);let n=new DataView(a.buffer,a.byteOffset,a.byteLength);for(;t<a.length-4;)if(n.getInt32(t,!0)===-1){if(t+=4,t+4>a.length)break;let s=n.getInt32(t,!0);for(t+=4+s;t+4<=a.length&&n.getInt32(t,!0)!==-1;){let o=n.getInt32(t,!0);o>=0&&o<1e7&&e.add(o),t+=4}}else t++;return e}function Hn(a){let e=new Set,t=new DataView(a.buffer,a.byteOffset,a.byteLength);if(a.length<8)return e;let n=t.getUint32(0,!0);if(n===12346||n===12347){let r=n===12347,s=4,i=t.getUint16(s,!0);s+=2;let o=s;s+=i*4;for(let c=0;c<i&&s<a.length;c++){let l=t.getUint16(o+c*4,!0),f=t.getUint16(o+c*4+2,!0)+1,u=l<<16;for(let h=0;h<f&&s+2<=a.length;h++){let d=t.getUint16(s,!0);e.add(u|d),s+=2}}}return e}async function qt(a,e){if(a._deletedRows.has(e))return a._deletedRows.get(e);let t=a._fragments[e];if(!t?.deletionFile){let i=new Set;return a._deletedRows.set(e,i),i}let{url:n,fileType:r,numDeletedRows:s}=t.deletionFile;console.log(`[LanceQL] Loading ${s} deletions from ${n} (${r})`);try{let i=await fetch(n);if(!i.ok){console.warn(`[LanceQL] Failed to load deletion file: ${i.status}`);let f=new Set;return a._deletedRows.set(e,f),f}let o=await i.arrayBuffer(),c=new Uint8Array(o),l;return r==="arrow"?l=qn(c):l=Hn(c),console.log(`[LanceQL] Loaded ${l.size} deleted rows for fragment ${e}`),a._deletedRows.set(e,l),l}catch(i){console.error("[LanceQL] Error loading deletion file:",i);let o=new Set;return a._deletedRows.set(e,o),o}}async function Ht(a,e,t,n=10,r=null,s={}){let{normalized:i=!0,workerPool:o=null,useIndex:c=!0,nprobe:l=20}=s,f=e;if(f<0)throw new Error("No vector column found in dataset");let u=t.length;if(!a.hasIndex())throw new Error("No IVF index found. Vector search requires an IVF index for efficient querying.");if(a._ivfIndex.dimension!==u)throw new Error(`Query dimension (${u}) does not match index dimension (${a._ivfIndex.dimension}).`);if(!a._ivfIndex.hasPartitionIndex)throw new Error("IVF partition index (ivf_partitions.bin) not found. Required for efficient search.");return await jn(a,t,n,f,l,r)}async function jn(a,e,t,n,r,s){let i=a._ivfIndex.findNearestPartitions(e,r),o=await a._ivfIndex.fetchPartitionData(i,a._ivfIndex.dimension,(w,p)=>{if(s){let y=p>0?w/p:0;s(Math.floor(y*80),100)}});if(!o||o.rowIds.length===0)throw new Error("IVF index not available. This dataset requires ivf_vectors.bin for efficient search.");let{rowIds:c,vectors:l,preFlattened:f}=o,u=e.length,h=f?l.length/u:l.length,d=new Float32Array(h),m=O();if(m.isAvailable()){let w=m.getMaxVectorsPerBatch(u);if(f)for(let p=0;p<h;p+=w){let y=Math.min(p+w,h),_=y-p,x=l.subarray(p*u,y*u);try{let v=await m.batchCosineSimilarity(e,x,!0,!0);if(v){d.set(v,p);continue}}catch{}if(a.lanceql?.batchCosineSimilarityFlat){let v=a.lanceql.batchCosineSimilarityFlat(e,x,u,!0);d.set(v,p)}else for(let v=0;v<_;v++){let A=v*u,I=0;for(let S=0;S<u;S++)I+=e[S]*x[A+S];d[p+v]=I}}else for(let p=0;p<h;p+=w){let y=Math.min(p+w,h),_=l.slice(p,y);try{let x=await m.batchCosineSimilarity(e,_,!0,!1);if(x){d.set(x,p);continue}}catch{}for(let x=0;x<_.length;x++){let v=_[x];if(!v||v.length!==u)continue;let A=0;for(let I=0;I<u;I++)A+=e[I]*v[I];d[p+x]=A}}}else if(f)for(let w=0;w<h;w++){let p=w*u,y=0;for(let _=0;_<u;_++)y+=e[_]*l[p+_];d[w]=y}else for(let w=0;w<h;w++){let p=l[w];if(!p||p.length!==u)continue;let y=0;for(let _=0;_<u;_++)y+=e[_]*p[_];d[w]=y}s&&s(90,100);let g=new Array(c.length);for(let w=0;w<c.length;w++)g[w]={index:c[w],score:d[w]};let b=Math.min(t,g.length);return Wn(g,b),s&&s(100,100),{indices:g.slice(0,b).map(w=>w.index),scores:g.slice(0,b).map(w=>w.score),usedIndex:!0,searchedRows:c.length}}function Wn(a,e){if(e>=a.length||e<=0)return;let t=0,n=a.length-1;for(;t<n;){let r=t+n>>1;a[r].score>a[t].score&&ie(a,t,r),a[n].score>a[t].score&&ie(a,t,n),a[r].score>a[n].score&&ie(a,r,n);let s=a[n].score,i=t;for(let o=t;o<n;o++)a[o].score>=s&&(ie(a,i,o),i++);if(ie(a,i,n),i===e-1)break;i<e-1?t=i+1:n=i-1}}function ie(a,e,t){let n=a[e];a[e]=a[t],a[t]=n}function jt(a){if(!a._schema)return-1;for(let e=0;e<a._schema.length;e++){let t=a._schema[e];if(t.name==="embedding"||t.name==="vector"||t.type==="fixed_size_list"||t.type==="list")return e}return a._schema.length-1}function je(a,e){let t=0;for(let n=0;n<a._fragments.length;n++){let r=a._fragments[n];if(e<t+r.numRows)return{fragmentIndex:n,localIndex:e-t};t+=r.numRows}return null}function $(a,e){let t=new Map;for(let n of e){let r=je(a,n);r&&(t.has(r.fragmentIndex)||t.set(r.fragmentIndex,{localIndices:[],globalIndices:[]}),t.get(r.fragmentIndex).localIndices.push(r.localIndex),t.get(r.fragmentIndex).globalIndices.push(n))}return t}async function We(a,{offset:e=0,limit:t=50,columns:n=null,_isPrefetch:r=!1}={}){let s=[],i=0;for(let m=0;m<a._fragments.length;m++){let g=a._fragments[m],b=i,w=i+g.numRows;if(w>e&&b<e+t){let p=Math.max(0,e-b),y=Math.min(g.numRows,e+t-b);s.push({fragmentIndex:m,localOffset:p,localLimit:y-p,globalStart:b+p})}if(i=w,i>=e+t)break}if(s.length===0)return{columns:[],columnNames:a.columnNames,total:a._totalRows};let o=s.map(async m=>{let b=await(await a.openFragment(m.fragmentIndex)).readRows({offset:m.localOffset,limit:m.localLimit,columns:n});return{...m,result:b}}),c=await Promise.all(o);c.sort((m,g)=>m.globalStart-g.globalStart);let l=[],f=c[0]?.result.columnNames||a.columnNames,u=n?n.length:a._numColumns;for(let m=0;m<u;m++){let g=[];for(let b of c)b.result.columns[m]&&g.push(...b.result.columns[m]);l.push(g)}let h={columns:l,columnNames:f,total:a._totalRows},d=e+t;return!r&&d<a._totalRows&&t<=100&&Yn(a,d,t,n),h}function Yn(a,e,t,n){let r=`${e}-${t}-${n?.join(",")||"all"}`;if(a._prefetchCache?.has(r))return;a._prefetchCache||(a._prefetchCache=new Map);let s=We(a,{offset:e,limit:t,columns:n,_isPrefetch:!0}).then(i=>{a._prefetchCache.set(r,i)}).catch(()=>{});a._prefetchCache.set(r,s)}async function Wt(a,e,t){let n=$(a,t),r=new Map,s=[];for(let[i,o]of n)s.push((async()=>{let l=await(await a.openFragment(i)).readStringsAtIndices(e,o.localIndices);for(let f=0;f<o.globalIndices.length;f++)r.set(o.globalIndices[f],l[f])})());return await Promise.all(s),t.map(i=>r.get(i)||null)}async function Kt(a,e,t){let n=$(a,t),r=new Map,s=[];for(let[i,o]of n)s.push((async()=>{let l=await(await a.openFragment(i)).readInt64AtIndices(e,o.localIndices);for(let f=0;f<o.globalIndices.length;f++)r.set(o.globalIndices[f],l[f])})());return await Promise.all(s),new BigInt64Array(t.map(i=>r.get(i)||0n))}async function Yt(a,e,t){let n=$(a,t),r=new Map,s=[];for(let[i,o]of n)s.push((async()=>{let l=await(await a.openFragment(i)).readFloat64AtIndices(e,o.localIndices);for(let f=0;f<o.globalIndices.length;f++)r.set(o.globalIndices[f],l[f])})());return await Promise.all(s),new Float64Array(t.map(i=>r.get(i)||0))}async function Qt(a,e,t){let n=$(a,t),r=new Map,s=[];for(let[i,o]of n)s.push((async()=>{let l=await(await a.openFragment(i)).readInt32AtIndices(e,o.localIndices);for(let f=0;f<o.globalIndices.length;f++)r.set(o.globalIndices[f],l[f])})());return await Promise.all(s),new Int32Array(t.map(i=>r.get(i)||0))}async function Jt(a,e,t){let n=$(a,t),r=new Map,s=[];for(let[i,o]of n)s.push((async()=>{let l=await(await a.openFragment(i)).readFloat32AtIndices(e,o.localIndices);for(let f=0;f<o.globalIndices.length;f++)r.set(o.globalIndices[f],l[f])})());return await Promise.all(s),new Float32Array(t.map(i=>r.get(i)||0))}var ve=new se,oe=class a{constructor(e,t){this.lanceql=e,this.baseUrl=t.replace(/\/$/,""),this._fragments=[],this._schema=null,this._totalRows=0,this._numColumns=0,this._onFetch=null,this._fragmentFiles=new Map,this._isRemote=!0,this._ivfIndex=null,this._deletedRows=new Map}static async open(e,t,n={}){let r=new a(e,t);r._requestedVersion=n.version||null;let s=n.version?`${t}@v${n.version}`:t;if(!n.skipCache){let o=await ve.get(s);o&&o.schema&&o.fragments&&(r._schema=o.schema,r._fragments=o.fragments,r._numColumns=o.schema.length,r._totalRows=o.fragments.reduce((c,l)=>c+l.numRows,0),r._version=o.version,r._columnTypes=o.columnTypes||null,r._fromCache=!0)}return r._fromCache||(await r._tryLoadSidecar()||await r._loadManifest(),ve.set(s,{schema:r._schema,fragments:r._fragments,version:r._version,columnTypes:r._columnTypes||null}).catch(()=>{})),await r._tryLoadIndex(),(n.prefetch??!1)&&r._fragments.length>0&&r._prefetchFragments(),r}async _tryLoadSidecar(){try{let e=`${this.baseUrl}/.meta.json`,t=await fetch(e);if(!t.ok)return!1;let n=await t.json();return!n.schema||!n.fragments?!1:(this._schema=n.schema.map(r=>({name:r.name,id:r.index,type:r.type})),this._fragments=n.fragments.map(r=>({id:r.id,path:r.data_files?.[0]||`${r.id}.lance`,numRows:r.num_rows,physicalRows:r.physical_rows||r.num_rows,url:`${this.baseUrl}/data/${r.data_files?.[0]||r.id+".lance"}`,deletionFile:r.has_deletions?{numDeletedRows:r.deleted_rows||0}:null})),this._numColumns=n.num_columns,this._totalRows=n.total_rows,this._version=n.lance_version,this._columnTypes=n.schema.map(r=>{let s=r.type;return s.startsWith("vector[")?"vector":s==="float64"||s==="double"?"float64":s==="float32"?"float32":s.includes("int")?s:s==="string"?"string":"unknown"}),!0)}catch{return!1}}_prefetchFragments(){let e=this._fragments.map((t,n)=>this.openFragment(n).catch(()=>null));Promise.all(e).catch(()=>{})}hasIndex(){return this._ivfIndex!==null&&this._ivfIndex.centroids!==null}async _tryLoadIndex(){try{this._ivfIndex=await j.tryLoad(this.baseUrl)}catch{this._ivfIndex=null}}async _loadManifest(){let e=null,t=0;if(this._requestedVersion){t=this._requestedVersion;let n=`${this.baseUrl}/_versions/${t}.manifest`,r=await fetch(n);if(!r.ok)throw new Error(`Version ${t} not found (${r.status})`);e=new Uint8Array(await r.arrayBuffer())}else{let n=[1,5,10,20,50,100],r=await Promise.all(n.map(async c=>{try{let l=`${this.baseUrl}/_versions/${c}.manifest`;return(await fetch(l,{method:"HEAD"})).ok?c:0}catch{return 0}})),s=Math.max(...r);if(s>0)for(let c=s+1;c<=s+50;c++)try{let l=`${this.baseUrl}/_versions/${c}.manifest`;if((await fetch(l,{method:"HEAD"})).ok)s=c;else break}catch{break}if(t=s,t===0)throw new Error("No manifest found in dataset");let i=`${this.baseUrl}/_versions/${t}.manifest`,o=await fetch(i);if(!o.ok)throw new Error(`Failed to fetch manifest: ${o.status}`);e=new Uint8Array(await o.arrayBuffer())}this._version=t,this._latestVersion=this._requestedVersion?null:t,this._parseManifest(e)}async listVersions(){let e=[],t=this._latestVersion||100;return(await Promise.all(Array.from({length:t},(r,s)=>s+1).map(async r=>{try{let s=`${this.baseUrl}/_versions/${r}.manifest`;return(await fetch(s,{method:"HEAD"})).ok?r:0}catch{return 0}}))).filter(r=>r>0)}get version(){return this._version}_parseManifest(e){let t=new DataView(e.buffer,e.byteOffset),n=t.getUint32(0,!0),r=4+n,s;if(r+4<e.length){let u=t.getUint32(r,!0);u>0&&r+4+u<=e.length?s=e.slice(r+4,r+4+u):s=e.slice(4,4+n)}else s=e.slice(4,4+n);let i=0,o=[],c=[],l=()=>{let u=0,h=0;for(;i<s.length;){let d=s[i++];if(u|=(d&127)<<h,(d&128)===0)break;h+=7}return u},f=u=>{if(u===0)l();else if(u===2){let h=l();i+=h}else u===5?i+=4:u===1&&(i+=8)};for(;i<s.length;){let u=l(),h=u>>3,d=u&7;if(h===1&&d===2){let m=l(),g=i+m,b=null,w=null,p=null;for(;i<g;){let y=l(),_=y>>3,x=y&7;if(x===0){let v=l();_===3&&(w=v)}else if(x===2){let v=l(),A=s.slice(i,i+v);i+=v,_===2?b=new TextDecoder().decode(A):_===5&&(p=new TextDecoder().decode(A))}else f(x)}b&&o.push({name:b,id:w,type:p})}else if(h===2&&d===2){let m=l(),g=i+m,b=null,w=null,p=0,y=null;for(;i<g;){let _=l(),x=_>>3,v=_&7;if(v===0){let A=l();x===1?b=A:x===4&&(p=A)}else if(v===2){let A=l(),I=s.slice(i,i+A);if(i+=A,x===2){let S=0;for(;S<I.length;){let U=I[S++],L=U>>3,B=U&7;if(B===2){let C=0,k=0;for(;S<I.length;){let T=I[S++];if(C|=(T&127)<<k,(T&128)===0)break;k+=7}let G=I.slice(S,S+C);S+=C,L===1&&(w=new TextDecoder().decode(G))}else if(B===0)for(;S<I.length&&(I[S++]&128)!==0;);else B===5?S+=4:B===1&&(S+=8)}}else x===3&&(y=this._parseDeletionFile(I,b))}else f(v)}if(w){let _=y?p-y.numDeletedRows:p;c.push({id:b,path:w,numRows:_,physicalRows:p,deletionFile:y,url:`${this.baseUrl}/data/${w}`})}}else f(d)}this._schema=o,this._fragments=c,this._numColumns=o.length,this._totalRows=c.reduce((u,h)=>u+h.numRows,0)}_parseDeletionFile(e,t){return Vt(e,t,this.baseUrl)}async _loadDeletedRows(e){return qt(this,e)}async isRowDeleted(e,t){return(await this._loadDeletedRows(e)).has(t)}get numColumns(){return this._numColumns}get rowCount(){return this._totalRows}async getRowCount(e=0){return this._totalRows}async readVectorAt(e,t){let n=this._getFragmentForRow(t);return n?await(await this.openFragment(n.fragmentIndex)).readVectorAt(e,n.localIndex):new Float32Array(0)}async getVectorInfo(e){if(this._fragments.length===0)return{rows:0,dimension:0};let n=await(await this.openFragment(0)).getVectorInfo(e);return n.dimension===0?{rows:0,dimension:0}:{rows:this._totalRows,dimension:n.dimension}}get columnNames(){return this._schema?this._schema.map(e=>e.name):[]}get schema(){return this._schema}get fragments(){return this._fragments}get size(){if(this._cachedSize)return this._cachedSize;let e=0;for(let t=0;t<(this._columnTypes?.length||0);t++){let n=this._columnTypes[t];if(n==="int64"||n==="float64"||n==="double")e+=8;else if(n==="int32"||n==="float32")e+=4;else if(n==="string")e+=50;else if(n==="vector"||n?.startsWith("vector[")){let r=n?.match(/\[(\d+)\]/),s=r?parseInt(r[1]):384;e+=s*4}else e+=8}return e===0&&(e=100),this._cachedSize=this._totalRows*e,this._cachedSize}onFetch(e){this._onFetch=e}async openFragment(e){if(e<0||e>=this._fragments.length)throw new Error(`Invalid fragment index: ${e}`);if(this._fragmentFiles.has(e))return this._fragmentFiles.get(e);let t=this._fragments[e],n=await W.open(this.lanceql,t.url);return this._onFetch&&n.onFetch(this._onFetch),this._fragmentFiles.set(e,n),n}async readRows(e={}){return We(this,e)}async detectColumnTypes(){if(this._columnTypes&&this._columnTypes.length>0)return this._columnTypes;if(this._fragments.length===0)return[];let t=await(await this.openFragment(0)).detectColumnTypes();this._columnTypes=t;let n=this._requestedVersion?`${this.baseUrl}@v${this._requestedVersion}`:this.baseUrl;return ve.get(n).then(r=>{r&&(r.columnTypes=t,ve.set(n,r).catch(()=>{}))}).catch(()=>{}),t}_getFragmentForRow(e){return je(this,e)}_groupIndicesByFragment(e){return $(this,e)}async readStringsAtIndices(e,t){return Wt(this,e,t)}async readInt64AtIndices(e,t){return Kt(this,e,t)}async readFloat64AtIndices(e,t){return Yt(this,e,t)}async readInt32AtIndices(e,t){return Qt(this,e,t)}async readFloat32AtIndices(e,t){return Jt(this,e,t)}async vectorSearch(e,t,n=10,r=null,s={}){return Ht(this,e,t,n,r,s)}_findVectorColumn(){return jt(this)}async executeSQL(e){let t=e.toUpperCase(),n=null,r=null,s=!1;if(t.includes("LIMIT")){let d=e.match(/LIMIT\s+(\d+)/i);d&&(n=parseInt(d[1]))}if(t.includes("OFFSET")){let d=e.match(/OFFSET\s+(\d+)/i);d&&(r=parseInt(d[1]))}if(t.includes("WHERE")&&(s=!0),!s&&t.includes("SELECT")&&(t.includes("*")||t.includes("SELECT *")))return await this.readRows({offset:r||0,limit:n||50});let i=this._fragments.map(async(d,m)=>{let g=await this.openFragment(m);try{return await g.executeSQL(e)}catch(b){return console.warn(`Fragment ${m} query failed:`,b),{columns:[],columnNames:[],total:0}}}),o=await Promise.all(i);if(o.length===0||o.every(d=>d.columns.length===0))return{columns:[],columnNames:this.columnNames,total:0};let c=o.find(d=>d.columns.length>0);if(!c)return{columns:[],columnNames:this.columnNames,total:0};let l=c.columns.length,f=c.columnNames,u=Array.from({length:l},()=>[]),h=0;for(let d of o){for(let m=0;m<l&&m<d.columns.length;m++)u[m].push(...d.columns[m]);h+=d.total}if(n){let d=r||0;for(let m=0;m<l;m++)u[m]=u[m].slice(d,d+n)}return{columns:u,columnNames:f,total:h}}close(){for(let e of this._fragmentFiles.values())e.close&&e.close();this._fragmentFiles.clear()}};var Jn=new TextEncoder,Hs=new TextDecoder,P,ae,z=0,K=0,Xt=()=>!z||!K?null:new Uint8Array(ae.buffer,z,K),Zt=a=>z&&a<=K?!0:(z&&P.free&&P.free(z,K),K=Math.max(a+1024,4096),z=P.alloc(K),z!==0),Xn=a=>{if(a instanceof Uint8Array)return Zt(a.length)?(Xt().set(a),[z,a.length]):[a];if(typeof a!="string")return[a];let e=Jn.encode(a);return Zt(e.length)?(Xt().set(e),[z,e.length]):[a]};var Zn=a=>({getVersion(){let e=P.getVersion(),t=e>>16&255,n=e>>8&255,r=e&255;return`${t}.${n}.${r}`},open(e){return new ee(a,e)},async openUrl(e){return await O().init(),await W.open(a,e)},async openDataset(e,t={}){return await O().init(),await oe.open(a,e,t)},parseFooter(e){let t=new Uint8Array(e),n=P.alloc(t.length);if(!n)return null;try{new Uint8Array(ae.buffer).set(t,n);let r=P.parseFooterGetColumns(n,t.length),s=P.parseFooterGetMajorVersion(n,t.length),i=P.parseFooterGetMinorVersion(n,t.length);return r===0&&s===0?null:{numColumns:r,majorVersion:s,minorVersion:i}}finally{P.free(n,t.length)}},isValidLanceFile(e){let t=new Uint8Array(e),n=P.alloc(t.length);if(!n)return!1;try{return new Uint8Array(ae.buffer).set(t,n),P.isValidLanceFile(n,t.length)===1}finally{P.free(n,t.length)}}}),Ke=class{static async load(e="./lanceql.wasm"){let n=await(await fetch(e)).arrayBuffer();P=(await WebAssembly.instantiate(n,{env:{opfs_open:(o,c)=>0,opfs_read:(o,c,l,f)=>0,opfs_size:o=>0n,opfs_close:o=>{},js_log:(o,c)=>{}}})).instance.exports,ae=P.memory;let s=null,i=new Proxy({},{get(o,c){return s||(s=Zn(i)),c in s?s[c]:c==="memory"?ae:c==="raw"||c==="wasm"?P:typeof P[c]=="function"?(...l)=>P[c](...l.flatMap(Xn)):P[c]}});return i}};var Ye=class{constructor(e){this.name=e,this._ready=!1}async open(){return this._ready?this:(await F("db:open",{name:this.name}),this._ready=!0,this)}async _ensureOpen(){this._ready||await this.open()}async createTable(e,t,n=!1){return await this._ensureOpen(),F("db:createTable",{db:this.name,tableName:e,columns:t,ifNotExists:n})}async dropTable(e,t=!1){return await this._ensureOpen(),F("db:dropTable",{db:this.name,tableName:e,ifExists:t})}async insert(e,t){return await this._ensureOpen(),F("db:insert",{db:this.name,tableName:e,rows:t})}async flush(){return await this._ensureOpen(),F("db:flush",{db:this.name})}async delete(e,t=null){return await this._ensureOpen(),F("db:delete",{db:this.name,tableName:e,where:t})}async update(e,t,n=null){return await this._ensureOpen(),F("db:update",{db:this.name,tableName:e,updates:t,where:n})}async select(e,t={}){await this._ensureOpen();let n={...t};return delete n.where,F("db:select",{db:this.name,tableName:e,options:n,where:t.whereAST||null})}async exec(e){return await this._ensureOpen(),F("db:exec",{db:this.name,sql:e})}async getTable(e){return await this._ensureOpen(),F("db:getTable",{db:this.name,tableName:e})}async listTables(){return await this._ensureOpen(),F("db:listTables",{db:this.name})}async compact(){return await this._ensureOpen(),F("db:compact",{db:this.name})}async*scan(e,t={}){await this._ensureOpen();let n=await F("db:scanStart",{db:this.name,tableName:e,options:t});for(;;){let{batch:r,done:s}=await F("db:scanNext",{db:this.name,streamId:n});if(r.length>0&&(yield r),s)break}}async close(){await this._ensureOpen(),await this.flush()}async listVersions(e){return await this._ensureOpen(),F("db:listVersions",{db:this.name,tableName:e})}async selectAtVersion(e,t,n={}){await this._ensureOpen();let r={...n};return delete r.where,F("db:selectAtVersion",{db:this.name,tableName:e,version:t,options:r,where:n.whereAST||null})}async restoreToVersion(e,t){return await this._ensureOpen(),F("db:restoreTable",{db:this.name,tableName:e,version:t})}};export{R as DistanceMetric,Ke as LanceQL,Ye as LocalDatabase,Q as PureLanceWriter,oe as RemoteLanceDataset,ue as TableRef,le as Vault,J as WritableDataFrame,tt as default,rt as getGPUAggregator,ft as getGPUGrouper,ot as getGPUJoiner,lt as getGPUSorter,Z as getGPUVectorSearch,Tn as opfsStorage,tt as vault};
//# sourceMappingURL=lanceql.esm.js.map

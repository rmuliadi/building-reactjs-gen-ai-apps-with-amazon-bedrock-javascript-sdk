import{B as n,l as u}from"./index-fd_rTpfv.js";class c extends n{static lc_name(){return"VectorDBQAChain"}get inputKeys(){return[this.inputKey]}get outputKeys(){return this.combineDocumentsChain.outputKeys.concat(this.returnSourceDocuments?["sourceDocuments"]:[])}constructor(e){super(e),Object.defineProperty(this,"k",{enumerable:!0,configurable:!0,writable:!0,value:4}),Object.defineProperty(this,"inputKey",{enumerable:!0,configurable:!0,writable:!0,value:"query"}),Object.defineProperty(this,"vectorstore",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),Object.defineProperty(this,"combineDocumentsChain",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),Object.defineProperty(this,"returnSourceDocuments",{enumerable:!0,configurable:!0,writable:!0,value:!1}),this.vectorstore=e.vectorstore,this.combineDocumentsChain=e.combineDocumentsChain,this.inputKey=e.inputKey??this.inputKey,this.k=e.k??this.k,this.returnSourceDocuments=e.returnSourceDocuments??this.returnSourceDocuments}async _call(e,t){if(!(this.inputKey in e))throw new Error(`Question key ${this.inputKey} not found.`);const i=e[this.inputKey],r=await this.vectorstore.similaritySearch(i,this.k,e.filter,t==null?void 0:t.getChild("vectorstore")),s={question:i,input_documents:r},o=await this.combineDocumentsChain.call(s,t==null?void 0:t.getChild("combine_documents"));return this.returnSourceDocuments?{...o,sourceDocuments:r}:o}_chainType(){return"vector_db_qa"}static async deserialize(e,t){if(!("vectorstore"in t))throw new Error("Need to pass in a vectorstore to deserialize VectorDBQAChain");const{vectorstore:i}=t;if(!e.combine_documents_chain)throw new Error("VectorDBQAChain must have combine_documents_chain in serialized data");return new c({combineDocumentsChain:await n.deserialize(e.combine_documents_chain),k:e.k,vectorstore:i})}serialize(){return{_type:this._chainType(),combine_documents_chain:this.combineDocumentsChain.serialize(),k:this.k}}static fromLLM(e,t,i){const r=u(e);return new this({vectorstore:t,combineDocumentsChain:r,...i})}}export{c as VectorDBQAChain};

from django.shortcuts import render
from django.views import View
from langchain.chains import RetrievalQA
from llm import get_gemini_instance
from .vectordb import get_vectordb_instance,clear_pinecone
from .helper import chain_type_kwargs, handle_uploaded_file, extract_text,text_split
import json
from django.http import JsonResponse
import os

# Create your views here.
llm = get_gemini_instance()
docsearch = get_vectordb_instance()
qa=RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs)

class HomeView(View):
    def get(self, request):
        return render(request, 'ask-vault/home.html')
    
    def post(self, request):
        try:
            data = json.loads(request.body.decode("utf-8"))  # Parse JSON request
            question = data.get('question')
            if not question:
                return JsonResponse({"error": "Please enter a question"}, status=400)

            result = qa({"query": question})
            # print(result)
            return JsonResponse({"answer": result["result"]})  # ✅ Return JSON response

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        
class UploadKnowledgeBase(View):
    context = dict()
    template_name = 'ask-vault/home.html'
    
    # def get(self, request):
    #     return render(request, self.template_name, context=self.context)
    
    def post(self, request):
        context_text = request.POST.get('context')
        context_file = request.FILES.get('contextFile')

        if not context_text and not context_file:
            self.context['message'] = "Please enter text or upload a file."
            return render(request, self.template_name, self.context)
        
        clear_pinecone()
        
        if context_text:
            # docsearch.upsert_documents([{"document": context_text}])
            docsearch.add_texts([context_text])
            self.context['message'] = "Text context uploaded successfully."

        if context_file:
            file_path = handle_uploaded_file(context_file)
            extracted_texts = extract_text(file_path, context_file.name)

            if extracted_texts:
                text_chunks = text_split(extracted_texts)
                text_strings = [chunk.page_content for chunk in text_chunks]
                docsearch.add_texts(text_strings)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(e)
                self.context['message'] = "File uploaded and stored in Pinecone."

        return render(request, self.template_name, self.context)

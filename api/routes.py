from flask import Blueprint, jsonify, request

from services.cache import CacheHandler
from services.embedding_model import TransformerHandler
from services.language_model import LanguageModelHandler


api = Blueprint('api', __name__)

@api.route('/ask', methods=['POST'])
def ask():
    input = request.form.get('input')
    
    embeddings = TransformerHandler().encode_input(input)
    
    output = CacheHandler().get_from_cache(embeddings)
    
    if (output and 
        output['matches'][0]['score'] >= 0.80):
        return jsonify(dict(response=output['matches'][0]['metadata']['answer'])), 200
    
    try:
        output = LanguageModelHandler().ask_language_model(input)
        CacheHandler.write_on_cache(input, output)
        
        return jsonify(dict(response=output)), 200
    
    except Exception as e:
        return jsonify(dict(response=e)), 500

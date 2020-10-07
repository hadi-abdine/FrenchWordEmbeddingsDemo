from flask import Flask, redirect, url_for, render_template, request, jsonify
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

app = Flask(__name__)
# modelw2v = gensim.models.Word2Vec.load('word2vec/dascim2.model', mmap='r')
modelw2v = KeyedVectors.load_word2vec_format("../word2vec/dascim2.bin", binary=True, limit=500000) # change to your embeddings

@app.route("/")
def home():
    return render_template("base.html")



@app.route("/resources")
def resources():
    return render_template("resources.html")


@app.route("/analogy", methods=['POST', 'GET'])
def analogy():
    word1 = request.form['word1']
    word2 = request.form['word2']
    word3 = request.form.get("word3", False) 
    if word1 not in modelw2v.vocab:
        if word1=='':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the second word is not in the vocabulary'
        return jsonify({'result' : 'success', 'word_4' : res})
    elif word2 not in modelw2v.vocab:
        if word2=='':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the first word is not in the vocabulary'
        return jsonify({'result' : 'success', 'word_4' : res})
    elif word3 not in modelw2v.vocab:
        if word3=='':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the third word is not in the vocabulary'
        return jsonify({'result' : 'success', 'word_4' : res})
    word4 = modelw2v.most_similar(positive=[word2, word3], negative=[word1])

    res = ''
    for word in word4:
        res = res + word[0] + ', ' + str(round(word[1], 3)) + '\n'
    return jsonify({'result' : 'success', 'word_4' : res})

@app.route("/similarityscore", methods=['POST', 'GET'])
def simscore():
    sim1 = request.form['sim1']
    sim2 = request.form['sim2']
    if sim1 not in modelw2v.wv.vocab:
        if sim1=='':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the first word is not in the vocabulary'
        return jsonify({'result' : 'success', 'simscore' : res})
    elif sim2 not in modelw2v.wv.vocab:
        if sim2=='':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the second word is not in the vocabulary'
        return jsonify({'result' : 'success', 'simscore' : res})

    res = str(modelw2v.similarity(sim1, sim2))

    return jsonify({'result' : 'success', 'simscore' : res})

@app.route("/similaritywords", methods=['POST', 'GET'])
def simwords():
    wordgoal = request.form['wordgoal']
    if wordgoal not in modelw2v.wv.vocab:
        if wordgoal=='':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the word is not in the vocabulary'
        return jsonify({'result' : 'success', 'simwords' : res})
    
    print(modelw2v) 
    print('toto')

    simwordslist = modelw2v.similar_by_word(wordgoal)

    res = ''
    for word in simwordslist :
        res = res + word[0] + ', ' + str(round(word[1], 3)) + '\n'

    return jsonify({'result' : 'success', 'simwords' : res})


if __name__ == "__main__":
    app.run(debug=True)



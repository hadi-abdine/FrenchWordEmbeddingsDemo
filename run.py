from flask import Flask, redirect, url_for, render_template, request, jsonify, send_from_directory
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os
from flask_htpasswd import HtPasswdAuth

app = Flask(__name__)
app.config['FLASK_HTPASSWD_PATH'] = '../word2vec2/demo.htpasswd'
htpasswd = HtPasswdAuth(app)
w2vpath = os.path.join(app.root_path, "../word2vec/dascim2.bin")
modelw2v = KeyedVectors.load_word2vec_format(w2vpath, binary=True, limit=500000) # change to your embeddings

@app.route("/")
def home():
    return render_template("base.html")



@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
@htpasswd.required
def download_file(filename, user):
    uploads = os.path.join(app.root_path,'../word2vec2/')
    return send_from_directory(uploads, filename)

@app.route("/analogy", methods=['POST', 'GET'])
def analogy():
    word1 = request.form['word1'].lower()
    word2 = request.form['word2'].lower()
    word3 = request.form.get("word3", False).lower()
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
    sim1 = request.form['sim1'].lower()
    sim2 = request.form['sim2'].lower()
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
    wordgoal = request.form['wordgoal'].lower()
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
    app.run()



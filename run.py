from flask import Flask, redirect, url_for, render_template, request, jsonify, send_from_directory
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os
from flask_htpasswd import HtPasswdAuth
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from torch.nn import Softmax
import pandas as pd

app = Flask(__name__)
app.config['FLASK_HTPASSWD_PATH'] = '../word2vec/demo.htpasswd'
htpasswd = HtPasswdAuth(app)
w2vpath = os.path.join(app.root_path, "../word2vec/dascim2.bin")
modelw2v = KeyedVectors.load_word2vec_format(
    w2vpath, binary=True, limit=500000)  # change to your embeddings
barthez_tokenizer_sum = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-abstract")
barthez_model_sum = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-abstract")
barthez_tokenizer_title = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-title")
barthez_model_title = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-title")
barthez_tokenizer_sentiment = AutoTokenizer.from_pretrained(
    "moussaKam/barthez")
barthez_model_sentiment = AutoModelForSequenceClassification.from_pretrained(
    "moussaKam/barthez-sentiment-classification")
barthez_model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez")
m = Softmax(dim=1)


@app.route("/")
def home():
    return render_template("base.html")


@app.route("/resources")
def resources():
    return render_template("resources.html")


@app.route("/barthez")
def barthezpage():
    return render_template("barthez.html")


@app.route("/frWordEmbeddings")
def frembdpage():
    return render_template("frembd.html")


@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
@htpasswd.required
def download_file(filename, user):
    uploads = os.path.join(app.root_path, '../word2vec2/')
    return send_from_directory(uploads, filename)


@app.route("/analogy", methods=['POST', 'GET'])
def analogy():
    word1 = request.form['word1'].lower()
    word2 = request.form['word2'].lower()
    word3 = request.form.get("word3", False).lower()
    if word1 not in modelw2v.vocab:
        if word1 == '':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the second word is not in the vocabulary'
        return jsonify({'result': 'success', 'word_4': res})
    elif word2 not in modelw2v.vocab:
        if word2 == '':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the first word is not in the vocabulary'
        return jsonify({'result': 'success', 'word_4': res})
    elif word3 not in modelw2v.vocab:
        if word3 == '':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the third word is not in the vocabulary'
        return jsonify({'result': 'success', 'word_4': res})
    word4 = modelw2v.most_similar(positive=[word2, word3], negative=[word1])

    res = ''
    for word in word4:
        res = res + word[0] + ', ' + str(round(word[1], 3)) + '\n'
    return jsonify({'result': 'success', 'word_4': res})


@app.route("/similarityscore", methods=['POST', 'GET'])
def simscore():
    sim1 = request.form['sim1'].lower()
    sim2 = request.form['sim2'].lower()
    if sim1 not in modelw2v.wv.vocab:
        if sim1 == '':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the first word is not in the vocabulary'
        return jsonify({'result': 'success', 'simscore': res})
    elif sim2 not in modelw2v.wv.vocab:
        if sim2 == '':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the second word is not in the vocabulary'
        return jsonify({'result': 'success', 'simscore': res})

    res = str(modelw2v.similarity(sim1, sim2))

    return jsonify({'result': 'success', 'simscore': res})


@app.route("/similaritywords", methods=['POST', 'GET'])
def simwords():
    wordgoal = request.form['wordgoal'].lower()
    if wordgoal not in modelw2v.wv.vocab:
        if wordgoal == '':
            res = 'Forgot to write a word?'
        else:
            res = 'ERROR : the word is not in the vocabulary'
        return jsonify({'result': 'success', 'simwords': res})

    print(modelw2v)
    print('toto')

    simwordslist = modelw2v.similar_by_word(wordgoal)

    res = ''
    for word in simwordslist:
        res = res + word[0] + ', ' + str(round(word[1], 3)) + '\n'

    return jsonify({'result': 'success', 'simwords': res})

@app.route("/barthezSum", methods=['POST', 'GET'])
def getsummary():
    input_text = request.json['fullText']
    input_ids = torch.tensor(
        [barthez_tokenizer_sum.encode(input_text, add_special_tokens=True)]
    )

    barthez_model_sum.eval()
    predict = barthez_model_sum.generate(input_ids, max_length=100)[0]


    abstract = str(barthez_tokenizer_sum.decode(predict, skip_special_tokens=True))

    return jsonify({'fullText' : input_text, 'abstract' : abstract})

@app.route("/barthezSumRating", methods=['POST', 'GET'])
def saveSumRating():
    input_text = request.json['fullText']
    output_summary = request.json['summary']
    label = request.json['label']
    data = {'text':  [input_text],
        'summary': [output_summary],
         'label': [label]
        }
    df = pd.DataFrame (data, columns = ['text','summary', 'label'])
    df.to_csv('summary.csv', mode='a', header=False)

    return jsonify({'fullText' : input_text, 'abstract' : output_summary})

@app.route("/barthezTitleRating", methods=['POST', 'GET'])
def saveTitleRating():
    input_text = request.json['fullText']
    output_summary = request.json['summary']
    label = request.json['label']
    data = {'text':  [input_text],
        'summary': [output_summary],
         'label': [label]
        }
    df = pd.DataFrame (data, columns = ['text','summary', 'label'])
    df.to_csv('title.csv', mode='a', header=False)

    return jsonify({'fullText' : input_text, 'abstract' : output_summary})

@app.route("/barthezTitle", methods=['POST', 'GET'])
def gettitle():
    input_text = request.json['fullText']
    input_ids = torch.tensor(
        [barthez_tokenizer_title.encode(input_text, add_special_tokens=True)])

    barthez_model_title.eval()
    predict = barthez_model_title.generate(input_ids, max_length=100)[0]

    title = str(barthez_tokenizer_title.decode(predict, skip_special_tokens=True))
    return jsonify({'fullText' : input_text, 'title' : title})

@app.route("/barthezSentiment", methods=['POST', 'GET'])
def getclass():
    input_text = request.json['fullText']
    if len(input_text):
        if input_text[-1] != '.':
            input_text += '.'
        input_ids = torch.tensor(
            [barthez_tokenizer_sentiment.encode(input_text, add_special_tokens=True)])

        predict = barthez_model_sentiment.forward(input_ids)[0]
        proba = m(predict)
        # p = str(proba[0][1].item() * 100) + "%"
        # n = str(proba[0][0].item() * 100) + "%"
        p = "{:.2f}".format(proba[0][1].item()*100)+"%"
        n = "{:.2f}".format(proba[0][0].item()*100)+"%"
    else:
        p = '0%'
        n = '0%'    

    return jsonify({'fullText': input_text, 'p': p, 'n': n})


@app.route("/barthezMLM", methods=['POST', 'GET'])
def getwords():
    input_text = request.json['fullText']
    if len(input_text):
        if input_text[-1] != '.':
            input_text += '.'
        input_ids = torch.tensor(
            [barthez_tokenizer_sentiment.encode(
                input_text, add_special_tokens=True)]
        )
        mask_idx = torch.where(input_ids == barthez_tokenizer_sentiment.mask_token_id)[
            1].tolist()[0]

        predict = barthez_model.forward(input_ids)[0]

        words = barthez_tokenizer_sentiment.decode(
            predict[:, mask_idx, :].topk(5).indices[0]).split()
        pbar = m(predict[:, mask_idx, :].topk(5).values)[0]
        probas = m(predict[:, mask_idx, :]).topk(5).values[0]

        s = []
        p = []
        for i in pbar:
            s.append(i.item())
        for i in probas:
            p.append(i.item())
        s1 = "{:.2f}".format(s[0]*100)+"%"
        s2 = "{:.2f}".format(s[1]*100)+"%"
        s3 = "{:.2f}".format(s[2]*100)+"%"
        s4 = "{:.2f}".format(s[3]*100)+"%"
        s5 = "{:.2f}".format(s[4]*100)+"%"
        p1 = "{:.2f}".format(p[0]*100)+"%"
        p2 = "{:.2f}".format(p[1]*100)+"%"
        p3 = "{:.2f}".format(p[2]*100)+"%"
        p4 = "{:.2f}".format(p[3]*100)+"%"
        p5 = "{:.2f}".format(p[4]*100)+"%"
        w1 = words[0]
        w2 = words[1]
        w3 = words[2]
        w4 = words[3]
        w5 = words[4]
    return jsonify({'fullText': input_text, 's1': s1, 's2':  s2, 's3': s3, 's4': s4, 's5': s5, 'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5})


if __name__ == "__main__":
    app.run(debug=True)

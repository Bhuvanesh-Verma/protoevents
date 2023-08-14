# streamlit_app.py
import streamlit as st
import spacy_streamlit
from collections import defaultdict

import pandas as pd
import spacy
from riveter import Riveter

@st.cache_data
def get_new_doc(text, riv_type):
    base_text = text
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe('sentencizer')
    docs = nlp(base_text)
    labels = []

    texts = []
    text_ids = []
    for i, sent in enumerate(docs.sents):
        texts.append(sent.text)
        text_ids.append(i)

    riveter = Riveter()
    if riv_type in ['effect', 'value']:
        riveter.load_rashkin_lexicon(riv_type)
    else:
        riveter.load_sap_lexicon(riv_type)
    riveter.train(texts,
                  text_ids)

    doc2ents = defaultdict(list)
    for t_id, text in zip(text_ids, texts):
        verbs = []
        doc = nlp(text)
        sub_verb = riveter.count_nsubj_for_doc(t_id)
        obj_verb = riveter.count_dobj_for_doc(t_id)
        for (sub, verb), val in sub_verb.items():
            verbs.append(verb)
        for (obj, verb), val in obj_verb.items():
            verbs.append(verb)
        ents = []
        for token in doc:
            label = None

            # Check if the token is a verb
            if token.text in verbs:
                label = "verb"

            if label is not None:
                ent = {
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "label": label
                }
                labels.append(label)
                ents.append(ent)
        for ent in doc.ents:
            ents.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
            labels.append(ent.label_)
        doc2ents[t_id].append({'text': text, 'ents': ents})
    labels = list(set(labels))

    doc_ents = []


    for t_id, ent_info in doc2ents.items():
        text = ent_info[0]['text']
        ents = ent_info[0]['ents']

        abs_tok = base_text.index(text)
        for ent in ents:
            start = abs_tok + ent['start']
            end = abs_tok + ent['end']
            label = ent['label']
            ent = {'start': start, 'end': end, 'label': label}
            doc_ents.append(ent)

    merged_doc = [{'text': base_text, 'ents': doc_ents, 'title':None}]

    return merged_doc, labels


#models = ["en_core_web_sm", "en_core_web_trf"]

st.title("Proto-events")
st.write("We use [Spacy](https://spacy.io/universe/project/video-spacys-ner-model-alt) to extract entities from the text and [Riveter](http://maartensap.com/pdfs/antoniak2023riveter.pdf) to extract verbs that as proxy for the event triggers.")

df = None
option =None
st.subheader('Select Data to analyse')
in_type = st.selectbox(
        '',
        (None,'Upload', 'EIT News Dataset'))
if in_type == 'Upload':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
elif in_type == 'EIT News Dataset':
    df = pd.read_csv('data/eit_news - EIT news.csv')
    option = 'body'
else:
    st.info("You haven't selected a data yet.")

if df is not None:

    if st.checkbox('Show raw data'):
        options = [None] + list(df.columns)
        st.subheader('Raw data')
        st.write(df)
        option = st.selectbox(
            'Select column containing text',
            (options))
    if option is not None:
        curr_doc = st.slider('Select any instance', 0, len(df), 0)
        text = df.loc[curr_doc, option]
        riv_name_2_id = {'Power': 'power', 'Agency': 'agency', 'Value': 'value', 'Effect': 'effect'}
        ops = [None]+ list(riv_name_2_id.keys())
        riv_type = st.selectbox(
        'Select Riveter Lexicon',
        (ops))
        if riv_type is not None:
            docs, labels = get_new_doc(text, riv_name_2_id[riv_type])
            spacy_streamlit.visualize_ner(docs, labels=labels, manual=True, show_table=False, )

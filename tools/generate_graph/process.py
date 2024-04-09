import json
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer

with open("process.json", "r") as f:
    js = json.load(f)
    organs = js["organs"]
    organ_disease = js["organ_disease"]
    disease_organ = js["disease_organ"]
    organ_synonym = js["organ_synonym"]
    disease_synonym = js["disease_synonym"]
    negative_list = js["negative_list"]
    positive_list = js["positive_list"]
    organs_tissue_description = js["organs_tissue_description"]
    general_disease = js["general_disease"]
# Set stopwords
stop_words = set(stopwords.words("english"))
# Stemming
ps = PorterStemmer()
# punctuation
punc = list(punctuation)


def get_reports(report_list):  # for IU-xray
    reports = []
    for i in range(len(report_list)):
        reports.append(report_list[i]["report"])
    return reports


def divide_to_sentences(reports):
    """
    This function is used to divide reports into several sentences.

    Args:
        reports: list[str], each str is a report

    Return:
        reports_sentences: list[list[str]], each list[str] is the divided sentences of one report
    """

    reports_sentences = []

    for report in reports:
        text_list = []

        text_new = parse_decimal(report)
        text_sentences = text_new.split(".")

        for sentence in text_sentences:
            if len(sentence) > 0:
                text_list.append(sentence)

        reports_sentences.append(text_list)

    return reports_sentences


def clean_sentence(reports):
    """
    This function is used to clean the reports.
    For example: This image doesn't show some diseases. --> This image does not show some diseases.
    """

    clean = []
    for report in reports:
        report_list = []

        for text in report:
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            # Keep letters only, and convet texts to lower case
            text = re.sub("[^a-z\s]", "", text.lower())
            # Remove punctuations
            text_nopunc = [char for char in text if char not in punc]
            text_nopunc = "".join(text_nopunc)
            wd = []
            for word in text_nopunc.split():
                wd.append(word)
            report_list.append(" ".join(wd))

        clean.append(report_list)

    return clean


def split_sentence(reports):
    """
    Split each sentence into a list of words.
    e.g.,  "a large hiatal hernia is noted" -> ['a', 'large', 'hiatal', 'hernia', 'is', 'noted', '.']
    """

    split_sen = []

    for report in reports:
        report_list = []

        for text in report:
            text_split = text.split()
            text_split.append(".")
            report_list.append(text_split)

        split_sen.append(report_list)

    return split_sen


def parse_decimal(text):
    """
    input: a sentence. e.g. "The size is 5.5 cm."
    return: a sentence. e.g. "The size is 5*5 cm."
    """

    find_float = lambda x: re.search("\d+(\.\d+)", x).group()
    text_list = []

    for word in text.split():
        try:
            decimal = find_float(word)
            new_decimal = decimal.replace(".", "*")
            text_list.append(new_decimal)
        except:
            text_list.append(word)

    return " ".join(text_list)


def pre_process(reports):
    reports_sentences = divide_to_sentences(reports)
    reports_clean = clean_sentence(reports_sentences)
    reports_split = split_sentence(reports_clean)

    return reports_split


def replace_synonym(sentence):
    sentence_list = sentence.split(" ")
    for i in range(len(sentence_list)):
        if sentence_list[i] in disease_synonym:
            sentence_list[i] = disease_synonym[sentence_list[i]]

    sentence = " ".join(sentence_list)

    return sentence, sentence_list


def replace_disease_synonym(sentence):
    sentence_list = sentence.split(" ")
    for i in range(len(sentence_list)):
        if sentence_list[i] in disease_synonym:
            sentence_list[i] = disease_synonym[sentence_list[i]]

    sentence = " ".join(sentence_list)

    return sentence, sentence_list


def delete_negative(sentence_organs, sentence_diseases, sentence):
    if len(sentence_diseases) == 0:
        return sentence_organs, sentence_diseases

    for negative_words in negative_list:
        if negative_words in sentence:
            ne_len = len(negative_words)
            app_index = sentence.find(negative_words)
            temp_sentence = sentence[ne_len + app_index:]

            temp_sentence_word_set = set(temp_sentence.split(" "))

            count = 0
            sentence_diseases_copy = sentence_diseases.copy()
            for disease in sentence_diseases_copy:
                disease_set = set(disease.split(" "))

                if disease_set.issubset(temp_sentence_word_set):
                    disease_index = sentence_diseases_copy.index(disease)

                    sentence_organs.pop(disease_index - count)

                    count += 1

    return sentence_organs, sentence_diseases


# This part is different from synonym substitution because diseases are described by more than one word


def associate_same_disease(sentence_diseases):
    if (
            "enlarge" in sentence_diseases
    ):  # enlarge a special case which is not in synonym substitution, but placed here
        idx = sentence_diseases.index("enlarge")
        sentence_diseases[idx] = "cardiomegaly"

    if "cardiomediastinal enlarged" in sentence_diseases:
        idx = sentence_diseases.index("cardiomediastinal enlarged")
        sentence_diseases[idx] = "cardiomegaly"

    if "pulmonary vascularity accentuate" in sentence_diseases:
        idx = sentence_diseases.index("pulmonary vascularity accentuate")
        sentence_diseases[idx] = "pulmonary vascularity increase"

    if "pulmonary vascularity prominent" in sentence_diseases:
        idx = sentence_diseases.index("pulmonary vascularity prominent")
        sentence_diseases[idx] = "pulmonary vascularity increase"

    return sentence_diseases


def remove_special_case(sentence_organs, sentence_diseases):
    if (
            "airspace consolidation" in sentence_diseases
            and "consolidation" in sentence_diseases
    ):
        idx = sentence_diseases.index("consolidation")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    if (
            "airspace hyperinflation" in sentence_diseases
            and "hyperinflation" in sentence_diseases
    ):
        idx = sentence_diseases.index("hyperinflation")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    if "airspace effusion" in sentence_diseases and "effusion" in sentence_diseases:
        idx = sentence_diseases.index("effusion")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    if (
            "granuloma calcification" in sentence_diseases
            and "granuloma" in sentence_diseases
    ):
        idx = sentence_diseases.index("granuloma")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    return sentence_organs, sentence_diseases


def return_negative(sentence_organs, sentence_diseases, sentence):
    if len(sentence_diseases) == 0:
        return sentence_organs, sentence_diseases

    for negative_words in negative_list:
        if negative_words in sentence:
            ne_len = len(negative_words)
            app_index = sentence.find(negative_words)
            temp_sentence = sentence[ne_len + app_index:]

            temp_sentence_word_set = set(temp_sentence.split(" "))

            count = 0
            sentence_diseases_copy = sentence_diseases.copy()
            for disease in sentence_diseases_copy:
                disease_set = set(disease.split(" "))

                if disease_set.issubset(temp_sentence_word_set):
                    disease_index = sentence_diseases_copy.index(disease)
                    negative_disease = "no " + sentence_diseases_copy[disease_index]
                    sentence_diseases[disease_index] = negative_disease

                    count += 1

    return sentence_organs, sentence_diseases


def find_diseases_organs(reports_list):
    organs_list = []
    diseases_list = []
    reports_replaced_list = []
    reports_replaced_split_list = []

    # step1: synonym substitution and split words
    for report in reports_list:
        report_replaced_list = []
        report_replaced_split_list = []

        for sentence in report:
            replaced_sentence, replaced_sentence_split = replace_synonym(sentence)
            report_replaced_list.append(replaced_sentence)
            report_replaced_split_list.append(replaced_sentence_split)

        reports_replaced_list.append(report_replaced_list)
        reports_replaced_split_list.append(report_replaced_split_list)

    # step2: find diseases and its corresponding organ
    for i in range(len(reports_replaced_list)):
        report = reports_replaced_list[i]
        report_word = reports_replaced_split_list[i]

        report_organs = []
        report_diseases = []

        for j in range(len(report)):
            sentence = report[j]
            sentence_word = report_word[j]
            sentence_word_set = set(sentence_word)

            sentence_organs = []
            sentence_diseases = []

            for key in disease_organ:
                key_set = set(key.split(" "))
                if key_set.issubset(sentence_word_set):
                    sentence_organs.append(disease_organ[key])
                    sentence_diseases.append(key)

            # step3: delete special airspace/granuloma
            sentence_organs, sentence_diseases = remove_special_case(
                sentence_organs, sentence_diseases
            )

            # step4: associate same diseases, e.g., cardiomegaly
            sentence_diseases = associate_same_disease(sentence_diseases)

            # step5: delete negative cases
            sentence_organs, sentence_diseases = return_negative(
                sentence_organs, sentence_diseases, sentence
            )

            report_organs.append(sentence_organs)
            report_diseases.append(sentence_diseases)

        organs_list.append(report_organs)
        diseases_list.append(report_diseases)

    return organs_list, diseases_list


def replace_organ_synonym(sentence):
    sentence_list = sentence.split(" ")
    for i in range(len(sentence_list)):
        if sentence_list[i] in organ_synonym:
            sentence_list[i] = organ_synonym[sentence_list[i]]

    sentence = " ".join(sentence_list)

    return sentence


def check_positive(sentence):
    key = ""

    # check whether contains normal description words
    for pos_word in positive_list:
        if pos_word in sentence:
            key += pos_word

    # if not, return key=[]
    if len(key) == 0:
        return []
    else:
        key += "-"

    # check corresponding organs or tissue
    for organ in organs_tissue_description:
        if organ in sentence:
            key += organ
            key += "_"
    if key[-1] == "_":
        key = key[:-1]

    # check whether have corresponding organs
    key_list = key.split("-")
    if len(key_list[1]) == 0:
        key += "other"
    return [key]


def check_no_negative(sentence):
    """
    This function is to set key for each sentence.

    Normal description key：positive-organ1_organ2

    Negative normal description key: no-disease1_organ1-disease2_organ2

    """

    key_list = []
    key = ""

    # iterate negative words
    for negative_words in negative_list:
        has_key = ""

        # if has negative words，further check diseases and organs
        if negative_words in sentence:
            # according to the position of negative words, get the temp sentence after the word
            ne_len = len(negative_words)
            app_index = sentence.find(negative_words)
            temp_sentence = sentence[ne_len + app_index:]

            # if the number of sentence after negative words, then the temp sentence is the whole sentence
            if len(temp_sentence) == 0:
                temp_sentence = sentence

            # get the set of words of the temp sentence
            temp_sentence_word_set = set(temp_sentence.split(" "))

            # iterate disease，if has disease，set key
            for disease in disease_organ:
                disease_set = set(disease.split(" "))
                if disease_set.issubset(temp_sentence_word_set):
                    if len(key) == 0:
                        key = "no-" + disease + "_" + disease_organ[disease]
                    else:
                        key = key + "-" + disease + "_" + disease_organ[disease]
                    has_key = "good"

            if len(key) > 0:
                key_list.append(key)

            # if find no diseases，iterate over organ
            # if the temp sentence has organ，then set key
            # if not, turn the whole temp sentence into the whole sentence and check organ
            if len(has_key) == 0:
                for organ in organs_tissue_description:
                    if organ in temp_sentence:
                        key = "no-None-" + organ
                        key_list.append(key)
                        has_key = "good"
                    else:
                        temp_sentence = sentence
                        if organ in temp_sentence:
                            key = "no-None-" + organ
                            key_list.append(key)
                            has_key = "good"

            # if find no organ, then iterate over general diseases
            # if the temp sentence has disease, set key
            if len(has_key) == 0:
                temp_sentence_word_set = set(temp_sentence.split(" "))
                for disease in general_disease:
                    disease_set = set(disease.split(" "))
                    if disease_set.issubset(temp_sentence_word_set):
                        key = "no-" + disease + "-None"
                        key_list.append(key)
                        has_key = "good"

            # if find no general disease，set key to no-unknown-unknown
            if len(has_key) == 0:
                key = "no-unknown-unknown"
                key_list.append(key)
    return key_list


def check_disease(df):
    # add disease column, add organs_with_disease column
    has_disease = []  # to add in dataframe
    disease_description_list = []  # overal

    organs_disease = []  # to add in dataframe
    organs_with_disease_list = []  # overal

    for item_idx in range(len(df)):
        item = df["diseases_list"][item_idx]
        item_organs = df["organs_list"][item_idx]
        item_organs_with_disease = []
        count_disease_num = 0

        for sentence_idx in range(len(item)):
            sentence = item[sentence_idx]
            sentence_organs = item_organs[sentence_idx]
            sentence_organs_with_disease = []

            for disease_idx in range(len(sentence)):
                disease = sentence[disease_idx]

                if isinstance(disease, str):
                    count_disease_num += 1
                    disease_description_list.append(disease)
                    try:
                        sentence_organs_with_disease.append(
                            sentence_organs[disease_idx]
                        )
                    except (IndexError, KeyError):
                        pass

            item_organs_with_disease.append(sentence_organs_with_disease)
        organs_disease.append(item_organs_with_disease)

        if count_disease_num == 0:
            has_disease.append(0)
        else:
            has_disease.append(count_disease_num)

    df["has_disease"] = has_disease  # has_disease

    for item in organs_disease:
        for sentence in item:
            if len(sentence) > 0:
                for organ in sentence:
                    organs_with_disease_list.append(organ)
    return disease_description_list, organs_with_disease_list


def count_sentence(df):
    count_sentence_has_disease = 0
    count_sentence_num = 0

    for item in df["diseases_list"].tolist():
        count_sentence_num += len(item)
        for sentence in item:
            for value in sentence:
                if isinstance(value, str):
                    count_sentence_has_disease += 1
                    continue
    return count_sentence_has_disease, count_sentence_num


# count each disease
def count_disease(disease_list):
    disease_description_count = {}

    for disease in disease_list:
        if disease not in disease_description_count:
            disease_description_count[disease] = 1
        else:
            disease_description_count[disease] += 1
    # sort the dict
    sorted_disease_description_count = {
        k: v
        for k, v in sorted(
            disease_description_count.items(), key=lambda item: item[1], reverse=True
        )
    }
    return sorted_disease_description_count


if __name__ == "__main__":
    df = pd.read_csv("output_mimic.csv")
    reports_list = list(df["findings"])
    reports_sentences = divide_to_sentences(reports_list)
    reports_clean = clean_sentence(reports_sentences)
    reports_split = split_sentence(reports_clean)
    split_list = [reports_split]
    clean_list = [reports_clean]
    bio_dictionary = {}

    for reports_split in split_list:
        for report in reports_split:
            for sentence in report:
                for word in sentence:
                    if word not in bio_dictionary:
                        bio_dictionary[word] = 1
                    else:
                        bio_dictionary[word] += 1
    bio_dictionary_remove_stopwords = {}
    for key in bio_dictionary:
        if key not in stopwords.words("english"):
            bio_dictionary_remove_stopwords[key] = bio_dictionary[key]

    sort_bio_dict = sorted(
        bio_dictionary_remove_stopwords.items(), key=lambda xs: xs[1], reverse=True
    )
    count_1_words = []
    for word, count in sort_bio_dict:
        if count == 1:
            count_1_words.append(word)
    for reports_split in split_list:
        for report in reports_split:
            for sentence in report:
                unique_word = []
                for word in sentence:
                    if word in count_1_words:
                        unique_word.append(word)

    df["split_by_sentence"] = reports_clean
    organs_list, diseases_list = find_diseases_organs(reports_clean)
    df["organs_list"] = organs_list
    df["diseases_list"] = diseases_list
    diseases_pool = {}
    for key in disease_organ:
        pool_key = key + "-" + disease_organ[key]
        diseases_pool[pool_key] = []
    disease_type = []

    for i in range(len(df)):
        item = df.iloc[i]
        organs = item["organs_list"]
        diseases = item["diseases_list"]
        sentences = item["split_by_sentence"]
        report_disease = []

        for j in range(len(sentences)):
            assert len(organs[j]) == len(diseases[j])
            organ = organs[j]
            disease = diseases[j]
            sentence = sentences[j]
            sentence_disease = []

            if len(organ) > 0:
                for t in range(len(organ)):
                    key = disease[t] + "-" + organ[t]
                    sentence_disease.append(key)
                    """if sentence not in diseases_pool[key]:
                        diseases_pool[key].append(sentence)"""
            report_disease.append(sentence_disease)
        disease_type.append(report_disease)
    df["disease_type"] = disease_type
    normal_list = []

    count = 0

    for i in range(len(df)):
        item = df.iloc[i]
        itme_organs = item["organs_list"]
        itme_diseases = item["diseases_list"]
        itme_sentences = item["split_by_sentence"]

        report_normal_list = []

        for j in range(len(itme_sentences)):
            if len(itme_organs[j]) == 0:
                cur_sentence, _ = replace_disease_synonym(itme_sentences[j])
                cur_sentence = replace_organ_synonym(cur_sentence)
                key = check_positive(
                    cur_sentence
                )  # if find no normal description words, return empty

                # if find no normal description, then check if has negative description
                """if len(key) == 0:
                    key = check_no_negative(cur_sentence)"""

                # mark the sentences which contains only one or two words

                """cur_sentence_split = cur_sentence.split(" ")
                if len(cur_sentence_split) == 1:
                    key = ["length_one"]
                elif len(cur_sentence_split) == 2:
                    key = ["length_two"]"""

                if len(key) == 0:
                    count += 1

            # set the key to an empty list if the sentence has diseases
            else:
                key = []

            report_normal_list.append(key)

        normal_list.append(report_normal_list)
    normal_pool = {}

    for i in range(len(normal_list)):
        item = df.iloc[i]
        normal_report = normal_list[i]
        diseases = item["diseases_list"]
        sentences = item["split_by_sentence"]

        for j in range(len(normal_report)):
            key_list = normal_report[j]
            for key in key_list:
                if key not in normal_pool:
                    normal_pool[key] = []

                if sentences[j] not in normal_pool[key]:
                    normal_pool[key].append(sentences[j])
    df["normal_type"] = normal_list
    df.to_csv("new_mimic.csv", header=True, index=False)
    print("processing finished!")

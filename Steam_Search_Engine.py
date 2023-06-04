#!/usr/bin/env python
# coding: utf-8

# Authors:
# 202011071 Onurcan Erenel
# 201911068 Ahmet Bugra Yaka

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    regexp_replace,
    array_contains,
    lower,
    udf,
    to_date,
)
from pyspark.sql.types import StringType, BooleanType, ArrayType, FloatType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.linalg import SparseVector, DenseVector
from numpy import dot, linalg
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import scrolledtext, StringVar


def create_spark_session():
    conf = SparkConf().setAppName("SteamAnalysis")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    return spark


def load_data(spark):
    df = spark.read.json("hdfs://localhost:9000/games.json")
    df = df.withColumn("name", regexp_replace(col("name"), "[^a-zA-Z0-9\s.,-_]", ""))
    return df


def tokenize_name(df):
    tokenizer = Tokenizer(inputCol="name", outputCol="words")
    return tokenizer.transform(df)


def is_string(inp):
    return isinstance(inp, str)


def create_string_udf():
    return udf(is_string, BooleanType())


def tokenize_description(df):
    df = df.filter(
        col("description").isNotNull() & create_string_udf()(col("description"))
    )
    tokenizer = Tokenizer(inputCol="description", outputCol="desc_words")
    df = tokenizer.transform(df)
    return df


def search_by_name(df, name):
    name = name.lower().split()
    for word in name:
        df = df.filter(array_contains(col("words"), word))
    return df


def fuzzy_filter(df, column, query, threshold=80):
    fuzzy_match_score = udf(
        lambda desc: fuzz.token_set_ratio(query, desc), IntegerType()
    )
    df = df.withColumn("fuzzy_match_score", fuzzy_match_score(df[column]))
    df_filtered = df.filter(df["fuzzy_match_score"] > threshold)
    return df_filtered


def search_by_genre(df, genre):
    genre = genre.lower()
    return df.filter(lower(df.genres).contains(genre))


def search_by_tag(df, tag):
    tag = tag.lower()
    return df.filter(lower(df.tags).contains(tag))


def search_by_categories(df, category):
    category = category.lower()
    return df.filter(lower(df.categories).contains(category))


def search_by_developer(df, developer):
    developer = developer.lower()
    return df.filter(lower(df.developers).contains(developer))


def search_by_publisher(df, publisher):
    publisher = publisher.lower()
    return df.filter(lower(df.publishers).contains(publisher))


def filter_by_price(df, min_price, max_price):
    return df.filter((df.current_price >= min_price) & (df.current_price <= max_price))


def filter_by_language(df, language):
    language = language.lower()
    return df.filter(lower(df.languages).contains(language))


def filter_by_platform(df, platform):
    platform = platform.upper()
    return df.filter(df.platforms.contains(platform))


def sort_by_rating(df):
    return df.orderBy(df.store_uscore.desc())


def sort_by_playtime(df):
    return df.orderBy(df.hltb_complete.desc())


def sort_by_popularity(df):
    return df.orderBy(df.igdb_popularity.desc())


def filter_by_release_date(df, start_date, end_date):
    return df.filter(
        (to_date(df.published_store) >= start_date)
        & (to_date(df.published_store) <= end_date)
    )


def filter_by_user_score(df, min_score, max_score):
    return df.filter((df.store_uscore >= min_score) & (df.store_uscore <= max_score))


def filter_by_voiceover_language(df, language):
    language = language.lower()
    return df.filter(lower(df.voiceovers).contains(language))


spark = create_spark_session()
df = load_data(spark)
df = tokenize_name(df)
df = tokenize_description(df)


def perform_search():
    name_query = name_entry.get()
    desc_query = desc_entry.get()
    genre_query = genre_entry.get()
    tag_query = tag_entry.get()
    category_query = category_entry.get()
    developer_query = developer_entry.get()
    publisher_query = publisher_entry.get()
    language_query = language_entry.get()
    platform_query = platform_entry.get()
    min_price = min_price_entry.get()
    max_price = max_price_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    min_score = min_score_entry.get()
    max_score = max_score_entry.get()
    voiceover_language_query = voiceover_language_entry.get()
    sort_option = sort_var.get()

    result = df
    if name_query:
        result = search_by_name(result, name_query)
    if desc_query:
        result = fuzzy_filter(result, "description", desc_query)
    if genre_query:
        result = search_by_genre(result, genre_query)
    if tag_query:
        result = search_by_tag(result, tag_query)
    if category_query:
        result = search_by_categories(result, category_query)
    if developer_query:
        result = search_by_developer(result, developer_query)
    if publisher_query:
        result = search_by_publisher(result, publisher_query)
    if language_query:
        result = filter_by_language(result, language_query)
    if platform_query:
        result = filter_by_platform(result, platform_query)
    if min_price and max_price:
        result = filter_by_price(result, float(min_price), float(max_price))
    if start_date and end_date:
        result = filter_by_release_date(result, start_date, end_date)
    if min_score and max_score:
        result = filter_by_user_score(result, float(min_score), float(max_score))
    if voiceover_language_query:
        result = filter_by_voiceover_language(result, voiceover_language_query)

    if sort_option == "Rating":
        result = sort_by_rating(result)
    elif sort_option == "Playtime":
        result = sort_by_playtime(result)
    elif sort_option == "Popularity":
        result = sort_by_popularity(result)

    result = result.select("name", "genres").limit(50)

    result_list = result.collect()

    result_text.delete("1.0", tk.END)

    for row in result_list:
        result_text.insert(
            tk.END,
            "Name: {}, Genres: {}\n".format(row["name"], ", ".join(row["genres"])),
        )

    result = result.select("name", "genres").limit(50)

    result_list = result.collect()

    result_text.delete("1.0", tk.END)
    for row in result_list:
        result_text.insert(
            tk.END, "{}\nGenres: {}\n\n".format(row["name"], "".join(row["genres"]))
        )


root = tk.Tk()
pop_vari = tk.IntVar()

# Widgets
name_label = tk.Label(root, text="Search by Name:")
name_entry = tk.Entry(root, width=50)
desc_entry = tk.Entry(root, width=50)
desc_label = tk.Label(root, text="Search by Description:")
genre_label = tk.Label(root, text="Search by Genre:")
genre_entry = tk.Entry(root, width=50)
tag_label = tk.Label(root, text="Search by Tag:")
tag_entry = tk.Entry(root, width=50)
category_label = tk.Label(root, text="Search by Category:")
category_entry = tk.Entry(root, width=50)
search_button = tk.Button(root, text="Search", command=perform_search)
result_text = scrolledtext.ScrolledText(root, width=70, height=30)
developer_label = tk.Label(root, text="Search by Developer:")
developer_entry = tk.Entry(root, width=50)
publisher_label = tk.Label(root, text="Search by Publisher:")
publisher_entry = tk.Entry(root, width=50)
language_label = tk.Label(root, text="Filter by Language:")
language_entry = tk.Entry(root, width=50)
platform_label = tk.Label(root, text="Filter by Platform:")
platform_entry = tk.Entry(root, width=50)
min_price_label = tk.Label(root, text="Minimum Price:")
min_price_entry = tk.Entry(root, width=50)
max_price_label = tk.Label(root, text="Maximum Price:")
max_price_entry = tk.Entry(root, width=50)
start_date_label = tk.Label(root, text="Start Date (YYYY-MM-DD):")
start_date_entry = tk.Entry(root, width=50)
end_date_label = tk.Label(root, text="End Date (YYYY-MM-DD):")
end_date_entry = tk.Entry(root, width=50)
min_score_label = tk.Label(root, text="Minimum User Score:")
min_score_entry = tk.Entry(root, width=50)
max_score_label = tk.Label(root, text="Maximum User Score:")
max_score_entry = tk.Entry(root, width=50)
voiceover_language_label = tk.Label(root, text="Filter by Voiceover Language:")
voiceover_language_entry = tk.Entry(root, width=50)
sort_label = tk.Label(root, text="Sort by:")
sort_var = tk.StringVar(root)
sort_var.set("No Sort")
sort_option_menu = tk.OptionMenu(
    root, sort_var, "No Sort", "Rating", "Playtime", "Popularity"
)


# Layout
name_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
name_entry.grid(row=0, column=1, padx=10, pady=10)
desc_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
desc_entry.grid(row=1, column=1, padx=10, pady=10)
genre_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
genre_entry.grid(row=2, column=1, padx=10, pady=10)
tag_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
tag_entry.grid(row=3, column=1, padx=10, pady=10)
category_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
category_entry.grid(row=4, column=1, padx=10, pady=10)
search_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)
developer_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
developer_entry.grid(row=4, column=1, padx=10, pady=10)
publisher_label.grid(row=5, column=0, padx=10, pady=10, sticky="e")
publisher_entry.grid(row=5, column=1, padx=10, pady=10)
language_label.grid(row=6, column=0, padx=10, pady=10, sticky="e")
language_entry.grid(row=6, column=1, padx=10, pady=10)
platform_label.grid(row=7, column=0, padx=10, pady=10, sticky="e")
platform_entry.grid(row=7, column=1, padx=10, pady=10)
min_price_label.grid(row=8, column=0, padx=10, pady=10, sticky="e")
min_price_entry.grid(row=8, column=1, padx=10, pady=10)
max_price_label.grid(row=9, column=0, padx=10, pady=10, sticky="e")
max_price_entry.grid(row=9, column=1, padx=10, pady=10)
start_date_label.grid(row=10, column=0, padx=10, pady=10, sticky="e")
start_date_entry.grid(row=10, column=1, padx=10, pady=10)
end_date_label.grid(row=11, column=0, padx=10, pady=10, sticky="e")
end_date_entry.grid(row=11, column=1, padx=10, pady=10)
min_score_label.grid(row=12, column=0, padx=10, pady=10, sticky="e")
min_score_entry.grid(row=12, column=1, padx=10, pady=10)
max_score_label.grid(row=13, column=0, padx=10, pady=10, sticky="e")
max_score_entry.grid(row=13, column=1, padx=10, pady=10)
voiceover_language_label.grid(row=14, column=0, padx=10, pady=10, sticky="e")
voiceover_language_entry.grid(row=14, column=1, padx=10, pady=10)
sort_label.grid(row=15, column=0, padx=10, pady=10, sticky="e")
sort_option_menu.grid(row=15, column=1, padx=10, pady=10)
search_button.grid(row=16, column=0, columnspan=2, padx=10, pady=10)
result_text.grid(row=17, column=0, columnspan=2)

root.mainloop()

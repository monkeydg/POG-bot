# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import argparse
import sys
from ..classes.stats import PlayerStat
from logging import getLogger

log = getLogger("pog_bot")

def generate_streamlit_url(player_stats): #, pog_stats):
    
    return("https://streamlit.io/welcome/")


def main():
    # see https://github.com/streamlit/streamlit/pull/450 and https://github.com/streamlit/streamlit/issues/337 for details around custom arg implementation and why we use two "--" before arguments
    parser = argparse.ArgumentParser(description='extracts player id from command line')
    parser.add_argument('--player_id', action='append', default=[], help="Enter the discord/player id. It should look something like this: '251201892372185088'")

    try:
        args = parser.parse_args()
        player_id = int(args.player_id)
        player_name = args.player_name
    except SystemExit as e:
        # This exception will be raised if an invalid command line argument is used. Streamlit doesn't exit gracefully so we have to force it to stop ourselves.
        log.warning("Invalid command line argument. Exiting Streamlit app.")
        print("ERROR!")
        os._exit(e.code)

    # if player id is blank, don't load this page in streamlit app and instead display just overall pog stats


    player_stats = PlayerStat.get_from_database(player_id, player_name)


    POG_FAVICON = "https://cdn.discordapp.com/emojis/739512629277753455.webp?size=96&quality=lossless"
    POG_LOGO = "https://media.discordapp.net/attachments/739231714554937455/739522071423614996/logo_png.png"
    POG_BANNER = "https://cdn.discordapp.com/attachments/786308579015655474/942073850861072414/Banner_Planetside_Open_Games.png"
    # Set page title and favicon.
    st.set_page_config(
        page_title="Planetside Open Games Statistics", page_icon=POG_FAVICON,
    )

    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown("test 123\n\ntestig45\ntest678")

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()

    # Display header.
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(POG_BANNER, width=600)


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    # @st.cache
    # def load_metadata():
    #     return pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

    # # This function uses some Pandas magic to summarize the metadata Dataframe.
    # @st.cache
    # def create_summary(metadata):
    #     one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
    #     summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
    #         "label_biker": "biker",
    #         "label_car": "car",
    #         "label_pedestrian": "pedestrian",
    #         "label_trafficLight": "traffic light",
    #         "label_truck": "truck"
    #     })
    #     return summary

    # # An amazing property of st.cached functions is that you can pipe them into
    # # one another to form a computation DAG (directed acyclic graph). Streamlit
    # # recomputes only whatever subset is required to get the right answer!
    # metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    # summary = create_summary(metadata)

    # # Uncomment these lines to peek at these DataFrames.
    # # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

    # # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    # selected_frame_index, selected_frame = frame_selector_ui(summary)
    # if selected_frame_index == None:
    #     st.error("No frames fit the criteria. Please select different label or number.")
    #     return

    # # Draw the UI element to select parameters for the YOLO object detector.
    # confidence_threshold, overlap_threshold = object_detector_ui()

    # # Load the image from S3.
    # image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    # image = load_image(image_url)

    # # Add boxes for objects on the image. These are the boxes for the ground image.
    # boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    # draw_image_with_boxes(image, boxes, "Ground Truth",
    #     "**Human-annotated data** (frame `%i`)" % selected_frame_index)
    pass

# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary):
    st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [10, 20])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

    # Draw an altair chart in the sidebar with information on the frame.
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x = "selected_frame")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255],
    }
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    pass

if __name__ == "__main__":
    main()
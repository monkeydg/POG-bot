# the goal is to avoid imports from the rest of the project where possible,
# so that we can eventually deploy the streamlit module on a separate instance with minimal refactoring

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import argparse
import sys
import asyncio
import jsonpickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from logging import getLogger
log = getLogger("pog_bot")

MATCH_LENGTH = 10  # todo: pull the length of each match half from config.cfg without integrating too heavily with the rest of the project.

# these values are hard-coded based on how the Planetside 2 census reports loadouts.
# We could pull these from config.py but for now we'll leave it hard-coded to keep interactive_stats.py more standalone from the rest of the project
LOADOUT_IDS_DICT = {
    "Infiltrator": [1, 8, 15],
    "Light Assault": [3, 10, 17],
    "Medic": [4, 11, 18],
    "Engineer": [5, 12, 19],
    "Heavy Assault": [6, 13, 20],
    "Max": [7, 14, 21]
}

class LoadoutStats:
    def __init__(self, l_id, data):
        self.id = l_id
        self.weight = data["weight"]
        self.kills = data["kills"]
        self.deaths = data["deaths"]
        self.net = data["net"]
        self.score = data["score"]

    @property
    def kdr(self):
        return self.kills / self.deaths

class PlayerStat:
    def __init__(self, player_id, name, data):
        self.id = player_id
        self.name = name
        self.matches = data["matches"]
        self.matches_won = data["matches_won"]
        self.matches_lost = data["matches_lost"]
        self.time_played = data["time_played"]
        self.times_captain = data["times_captain"]
        self.pick_order = data["pick_order"]
        self.loadouts = dict()
        for l_data in data["loadouts"].values():
            l_id = l_data["id"]
            self.loadouts[l_id] = LoadoutStats(l_id, l_data)
        self.loadout_scores = self.generate_loadout_scores_dict()
        self.loadout_kdrs = self.generate_loadout_kdrs_dict()
        
    def merge_loadout_ids(self, unmerged_dict):
        """
        Given a dictionary with loadout ids as the key, it merges loadouts together where the id represents the same loadout, and adds the values together"""
        merged = {
        "Infiltrator": 0,
        "Light Assault": 0,
        "Medic": 0,
        "Engineer": 0,
        "Heavy Assault": 0,
        "Max": 0
        }

        # as can be seen in loadout_ids_dict, there are some ids that correspond to the same loadout, so we need to merge them.
        for id in unmerged_dict:
            for name in LOADOUT_IDS_DICT:
                if id in LOADOUT_IDS_DICT[name]:
                    merged[name] += unmerged_dict[id]

        return merged


    def generate_loadout_scores_dict(self):
        """
        Generates a dictionary of loadout scores for the player. 
        The dictionary key is the loadout name (e.g. "Medic") and the value is the score.
        """
        scores_list = [loadout.score for loadout in self.loadouts.values()]
        ids_list = [loadout.id for loadout in self.loadouts.values()]
        scores_dict_unmerged = dict(zip(ids_list, scores_list)) 

        return self.merge_loadout_ids(scores_dict_unmerged)
        

    def generate_loadout_kdrs_dict(self):
        """
        Generates a dictionary of kill/death ratios for each loadout for the player. 
        The dictionary key is the loadout name (e.g. "Medic") and the value is the kdr.
        """
        kdr_list = [loadout.kills/loadout.deaths for loadout in self.loadouts.values()]
        ids_list = [loadout.id for loadout in self.loadouts.values()]
        kdrs_dict_unmerged = dict(zip(ids_list, kdr_list))

        return self.merge_loadout_ids(kdrs_dict_unmerged)

    @property
    def num_matches_played(self):
        return len(self.matches)

    @property
    def kills_per_match(self):
        return self.kpm * MATCH_LENGTH * 2

    @property
    def kpm(self):
        if self.time_played == 0:
            return 0
        return self.kills / self.time_played

    @property
    def cpm(self): # number of times captain on average
        return self.times_captain / self.num_matches_played

    @property
    def score(self):
        score = 0
        for loadout in self.loadouts.values():
            score += loadout.score
        return score

    @property
    def kills(self):
        kills = 0
        for loadout in self.loadouts.values():
            kills += loadout.kills
        return kills

    @property
    def deaths(self):
        deaths = 0
        for loadout in self.loadouts.values():
            deaths += loadout.deaths
        return deaths

    @property
    def net(self):
        net = 0
        for loadout in self.loadouts.values():
            net += loadout.score
        return net

    @property
    def kdr(self):
        return self.kills / self.deaths


def get_color(score):
    """
    Returns a color for css highlighting based on whether a not a score is good.
    A high performing score is above 1, medium is between 0.5 and 1, and low is below 0.5."""
    if score > 1:
        return "green"
    elif score > 0.5:
        return "orange"
    else:
        return "red"

def activate_css():
    """
    Define new CSS classes for custom text highlighting."""
    st.markdown("""
    <style>
    .h1 {
        font-size:30px !important;
    }
    .bold {
        font-weight: bold !important;
    }
    .green {
        color: #006e0f !important;
    }
    .red {
        color: #8f1800 !important;
    }
    .orange {
        color: #db8700 !important;
    }
    </style>
    """, unsafe_allow_html=True)  # we could put this into a styles.css file and import it, but with only a handful of css classes this is simpler.

async def main():
    # Parsing CLI arguments to get data for the streamlit app
    # see https://github.com/streamlit/streamlit/pull/450 and https://github.com/streamlit/streamlit/issues/337 
    # for details around custom arg implementation and why we use two sets of "--" when building arguments in the __init__ method of a StreamlitApp class
    parser = argparse.ArgumentParser(description='extracts player data from command line')
    parser.add_argument('--player_id', action='append', default=[], help="Discord/player id. It should look something like this: 251201892372185088")
    parser.add_argument('--player_name', action='append', default=[], help="Discord nickname")
    parser.add_argument('--player_stats', action='append', default=[], help="JSON representation of a PlayerStat object containing data about the given player")

    try:
        args = parser.parse_args() # get player id, name, and stats from CLI arguments
        data = jsonpickle.decode(args.player_stats[0])
        player_stats = PlayerStat(int(args.player_id[0]), args.player_name[0], data) # initiate a new PlayerStats object with the input from CLI arguments
    except SystemExit as e:
        # This exception will be raised if an invalid command line argument is used. Streamlit doesn't exit gracefully so we have to force it to stop ourselves.
        log.warning("Invalid command line argument. Exiting Streamlit app.")
        os._exit(e.code)  # exists streamlit subprocess without disrupting the main discord bot process

    # if player id is blank, don't load this page in streamlit app and instead display just overall pog stat
    
    ### STREAMLIT PAGE SETUP AND DATA VISUALIZATIONS ###
    POG_FAVICON = "https://cdn.discordapp.com/emojis/739512629277753455.webp?size=96&quality=lossless"
    POG_LOGO = "https://media.discordapp.net/attachments/739231714554937455/739522071423614996/logo_png.png"
    POG_BANNER = "https://cdn.discordapp.com/attachments/786308579015655474/942073850861072414/Banner_Planetside_Open_Games.png"

    # Set page title and favicon.
    st.set_page_config(
        page_title=f"{player_stats.name}'s POG Statistics", 
        page_icon=POG_FAVICON
    )
    activate_css()

    st.image(POG_BANNER)
    st.title(f"{player_stats.name}'s POG Statistics")
    st.caption(f"Discord id: {player_stats.id}  |  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # I intentionally acronyms as subheaders because it's much more common in our community 
    # KPM = kills per minute, KDR = kill/death ratio
    # I'm not a fan of writing HTML in python but there seems to be no pythonic way to customize streamlit styling
    # we have to use spans to ensure that text is printed across the same line
    st.markdown(f"<p class='h1'>Matches played: <span class='bold'>{player_stats.num_matches_played}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='h1'>Time played: <span class='bold'>{timedelta(minutes=player_stats.time_played)}", unsafe_allow_html=True)
    st.markdown(f"<p class='h1'>KDR: <span class='bold {get_color(player_stats.kdr)}'>{round(player_stats.kdr, 2)}</span> | \
        KPM: <span class='bold {get_color(player_stats.kpm)}'>{round(player_stats.kpm, 2)}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='h1'>% of matches as captain: <span class='bold'>{round(player_stats.cpm, 2)}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='h1'>Average kills per match: <span class='bold'>{round(player_stats.kills_per_match, 2)}</span></p>", unsafe_allow_html=True)
    

    df_loadout_scores = pd.DataFrame.from_dict(player_stats.loadout_scores, orient='index', columns=['Score'])
    st.header("Total Score per Loadout")
    st.bar_chart(df_loadout_scores)

    df_loadout_kdrs = pd.DataFrame.from_dict(player_stats.loadout_kdrs, orient='index', columns=['KDR'])
    st.header("KDR per Loadout")
    st.bar_chart(df_loadout_kdrs)


    fig1, ax1 = plt.subplots()
    ax1.pie([player_stats.matches_won, player_stats.matches_lost], labels=["Wins", "Losses"], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

    

    # we use a numpy matrix of 1s and 0s to create a visualization of games played vs not played,
    # where a colored square of the matrix indicates a match played a gray square indicates not played.
    # it's a bit like a waffle chart, but simpler since it doesn't need to group values
    win_loss_matrix = np.zeros(max(player_stats.matches) + 1)
    
    for match in player_stats.matches:  # one-hot encoding
        win_loss_matrix[match] = 1
    

    st.header("Matches Played vs Not Played")
    st.markdown("Last 1024 matches")
    # because we want to display a square matrix, we drop values at the start of the array. We can use a 32x32 matrix.
    # this is fine because we don't have match data before 569 anyways, and the graph isn't designed to be perfect,
    # just a quick visual representation of matches played vs not played recently
    grid_size = 32
    win_loss_matrix = win_loss_matrix[-(grid_size*grid_size):]  # select only the most recent n*n matches, so we can display a square matrix. Here we choose 32*32 = 1024.
    win_loss_matrix = np.reshape(win_loss_matrix, (grid_size, grid_size))

    # Define colormap for matches played vs not played
    cmapmine = ListedColormap(['grey', 'lawngreen'], N=2)

    # Plot the matrix
    win_loss_fig = plt.figure()
    plt.imshow(win_loss_matrix, cmap=cmapmine, vmin=0, vmax=1)
    win_loss_fig.axes[0].set_xticks([]) # remove x axis labels
    win_loss_fig.axes[0].set_yticks([]) # remove y axis labels
    st.pyplot(win_loss_fig)

    st.markdown(player_stats.pick_order.keys())
    player_stats.pick_order.values()

    plt.rcdefaults()
    fig, ax = plt.subplots()

    st.header("Pick order")
    # Example data
    pick_order_index = range(2, 12)
    # we can't use player_stats.pick_order.keys() to get the pick order because it sometimes has values outside 2-6. 
    # values of 0 and 1 in the keys are for captain, and values above 12 occur if there are substitutions mid-match. 
    # to make this graph cleaner, we'll only include the first 10 picks, and start the index at 2 to ignore the captain slots 0 and 1.
    
    y_pos = np.arange(len(pick_order_index))
    pick_order = []
    for label in pick_order_index:
        if str(label) in player_stats.pick_order.keys(): 
            pick_order.append(player_stats.pick_order[str(label)])
        else:
            # if a player was never picked in a given slot, set number of picks for that slot to 0
            pick_order.append(0)

    # For people viewing the graph, we are going to start the labels at 1 which means first pick
    # this is much more intuitive for an end user who doesn't know/care that 0 and 1 are for captains
    labels = range(1, 11)

    ax.barh(y_pos, pick_order, align='center')
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Number of times picked')

    st.pyplot(fig)

    st.markdown(args.player_stats[0])
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
    # we run this asynchronously to avoid blocking the main thread when starting streamlit.
    # Also, if we ever want to integrate class methods from stats.py, we'll need to await the function calls.
    asyncio.run(main()) 
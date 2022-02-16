"""
STREAMLIT APP FOR PLANETSIDE OPEN GAMES
Development can be followed at https://github.com/monkeydg/POG-bot/tree/stats
This is a Streamlit app that displays statistics about a player in Planetside Open Games.
Planetside Open Games, or POG, is a 6v6 matchmaking bot for the video game Planetside 2
The bot is built around discord, with integrations into the game's API, Mongodb, and Teamspeak 3

the goal is to avoid imports from the rest of the project where possible, so that we can
later deploy the streamlit module on a separate ec2 instance with minimal refactoring
"""

#import os
import sys
import argparse
import asyncio
import json
import pickle
import urllib
from datetime import datetime, timedelta
from logging import getLogger
import jsonpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

log = getLogger("pog_bot")

# If we integrate this code into the rest of the pog bot, we can pull the length of
# each match half from config.cfg instead of declaring it here
MATCH_LENGTH = 10

# these values are hard-coded based on how the Planetside 2 census reports loadouts.
# We could pull these from config.py but for now we'll leave it hard-coded,
# to keep interactive_stats.py more standalone from the rest of the project.
LOADOUT_IDS_DICT = {
    "Infiltrator": [1, 8, 15],
    "Light Assault": [3, 10, 17],
    "Medic": [4, 11, 18],
    "Engineer": [5, 12, 19],
    "Heavy Assault": [6, 13, 20],
    "Max": [7, 14, 21]
}

# Image/logo assets
POG_FAVICON = "https://cdn.discordapp.com/emojis/739512629277753455.webp?size=96&quality=lossless"
POG_LOGO = "https://i.imgur.com/LKDdmlf.png"
POG_BANNER = "https://i.imgur.com/YJPXSya.png"

class LoadoutStats:
    """
    Class used to track a loadout's statistics in a pythonic way instead of a dictionary.
    """
    def __init__(self, loadout_id, data):
        self.loadout_id = loadout_id
        self.weight = data["weight"]
        self.kills = data["kills"]
        self.deaths = data["deaths"]
        self.net = data["net"]
        self.score = data["score"]

    @property
    def kdr(self):
        """ Returns a loadout's kill/death ratio. """
        return self.kills / self.deaths

    # I don't actually use this property in the project
    # but I have to keep pylint happy for that 10/10 score...
    @property
    def kills_per_weight(self):
        """ Returns a loadout's kills per weight ratio. """
        return self.kills / self.weight

class PlayerStat:
    """
    Class used to track a player's statistics in a pythonic way instead of a dictionary.
    The properties of this class are calculated from the data in the dictionary used to
    initialize an instance of the class.
    """
    def __init__(self, player_id, data):
        self.player_id = player_id
        # had to remove this line below to make pylint happy and use
        # silly workarounds later in the code
        # self.name = name

        # this is only here because I was forced to move
        # attributes into class properties for lint:
        self.data = data
        self.matches = data["matches"]
        self.time_played = data["time_played"]
        self.times_captain = data["times_captain"]
        self.pick_order = data["pick_order"]
        self.loadouts = {}
        for loadout_data in data["loadouts"].values():
            loadout_id = loadout_data["id"]
            self.loadouts[loadout_id] = LoadoutStats(loadout_id, loadout_data)

    def generate_loadout_scores_dict(self):
        """
        Generates a dictionary of loadout scores for the player.
        The dictionary key is the loadout name (e.g. "Medic") and the value is the score.
        """
        scores_list = [loadout.score for loadout in self.loadouts.values()]
        ids_list = [loadout.loadout_id for loadout in self.loadouts.values()]
        scores_dict_unmerged = dict(zip(ids_list, scores_list))

        return merge_loadout_ids(scores_dict_unmerged)


    def generate_loadout_kdrs_dict(self):
        """
        Generates a dictionary of kill/death ratios for each loadout for the player.
        The dictionary key is the loadout name (e.g. "Medic") and the value is the kdr.
        """
        kdr_list = [loadout.kills/loadout.deaths for loadout in self.loadouts.values()]
        ids_list = [loadout.loadout_id for loadout in self.loadouts.values()]
        kdrs_dict_unmerged = dict(zip(ids_list, kdr_list))

        return merge_loadout_ids(kdrs_dict_unmerged)

    @property
    def loadout_scores(self):
        """
        In order to make pylint happy I had to move instance attributes into properties -_-
        Returns a dictionary of loadout scores for the player.
        """
        return self.generate_loadout_scores_dict()

    @property
    def loadout_kdrs(self):
        """
        In order to make pylint happy I had to move instance attributes into properties -_-
        Returns a dictionary of loadout KDRs for the player.
        """
        return self.generate_loadout_kdrs_dict()

    @property
    def matches_won(self):
        """
        In order to make pylint happy I had to move instance attributes into properties -_-
        Returns number of matches won.
        """
        return self.data["matches_won"]

    @property
    def matches_lost(self):
        """
        In order to make pylint happy I had to move instance attributes into properties -_-
        Returns number of matches lost.
        """
        return self.data["matches_lost"]

    @property
    def num_matches_played(self):
        """ Returns the player's total number of matches played """
        return len(self.matches)

    @property
    def kills_per_match(self):
        """ Returns the player's average kills per match """
        return self.kpm * MATCH_LENGTH * 2

    @property
    def kpm(self):
        """ Returns the player's kills per minute """
        if self.time_played == 0:
            return 0
        return self.kills / self.time_played

    @property
    def cpm(self): # number of times captain on average
        """ Returns the player's number of times as captain for matches played """
        return self.times_captain / self.num_matches_played

    @property
    def score(self):
        """
        Returns the player's total score
        (see POG ruleset for details on calculation)
        """
        score = 0
        for loadout in self.loadouts.values():
            score += loadout.score
        return score

    @property
    def kills(self):
        """ Returns the player's total kills """
        kills = 0
        for loadout in self.loadouts.values():
            kills += loadout.kills
        return kills

    @property
    def deaths(self):
        """ Returns the player's total deaths """
        deaths = 0
        for loadout in self.loadouts.values():
            deaths += loadout.deaths
        return deaths

    @property
    def net(self):
        """ Returns the player's net score, where net = kills - deaths """
        net = 0
        for loadout in self.loadouts.values():
            net += loadout.score
        return net

    @property
    def kdr(self):
        """ Returns the player's kill/death ratio """
        return self.kills / self.deaths


def merge_loadout_ids(unmerged_dict):
    """
    Given a dictionary with loadout ids as the key, it merges loadouts together
    where the id represents the same loadout, and adds the values together
    """
    merged = {
    "Infiltrator": 0,
    "Light Assault": 0,
    "Medic": 0,
    "Engineer": 0,
    "Heavy Assault": 0,
    "Max": 0
    }

    # as can be seen in loadout_ids_dict, there are some ids that correspond to the same loadout
    # so we need to merge them.
    for loadout_id in unmerged_dict:
        for name in LOADOUT_IDS_DICT:
            if loadout_id in LOADOUT_IDS_DICT[name]:
                merged[name] += unmerged_dict[loadout_id]

    return merged

def dump_pkl(json_file, file_path="player_data.pkl"):
    """
    This is only used for dumping a json file for standalone testing/debugging
    """
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(json_file, pkl_file)

def fetch_pkl(file_path="player_data.pkl"):
    """
    This is only used for fetching a pkl file with json object
    for standalone testing/debugging
    """
    with open(file_path, 'rb') as pkl_file:
        pkl_file = pickle.load(pkl_file)

    return pkl_file

def get_color(score):
    """
    Returns a color for css highlighting based on whether a not a score is good.
    A high performing score is above 1, medium is between 0.5 and 1, and low is below 0.5.
    """
    if score > 1:
        return "green"
    if score > 0.5:
        return "orange"

    return "red"

def get_max_grid_size(array):
    """
    Returns the max grid side length based on the number of matches in a 1*n array of matches.
    We can't display more than there is data and we're missing data for the first 569 matches,
    so we omit these also.
    """
    max_side_length = np.sqrt(len(array) - 569)
    return int(max_side_length)

def activate_css():
    """
    Define new CSS classes for custom text highlighting in Streamlit.
    """
    # we could put this into a styles.css file and import it,
    # but with only a handful of css classes this is simpler.
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
    """, unsafe_allow_html=True)

def parse_cli():
    """
    Parses the command line arguments and returns a PlayerStat object that was
    extracted from JSON in the CLI arguments.
    See https://github.com/streamlit/streamlit/pull/450 and
    https://github.com/streamlit/streamlit/issues/337 for details around
    custom arg implementation and why we use two sets of "--" when building
    arguments in the __init__ method of a StreamlitApp class
    """
    parser = argparse.ArgumentParser(description='extracts player data from command line')
    parser.add_argument(
        '--player_id',
        action='append',
        default=[],
        help="Discord/player id. It should look something like this: 251201892372185088"
        )
    parser.add_argument(
        '--player_name',
        action='append',
        default=[],
        help="Discord nickname"
        )
    parser.add_argument(
        '--player_stats',
        action='append',
        default=[],
        help="JSON representation of a PlayerStat object containing data about the given player"
        )

    try:
        args = parser.parse_args() # get player id, name, and stats from CLI arguments
        json_data = args.player_stats[0]
        data = jsonpickle.decode(json_data)

        # initiate a new PlayerStats object with the input from CLI arguments
        player_stats = PlayerStat(int(args.player_id[0]), data)
    except SystemExit:
        # This exception will be raised if an invalid command line argument is used.
        # Streamlit doesn't exit gracefully so we have to force it to stop ourselves.
        # This exists streamlit subprocess without disrupting the main discord bot process.
        log.warning("Invalid command line argument. Exiting Streamlit app.")
        sys.exit()
        # this is better but lint gets upset accessing a protected member:
        # os._exit(exception.code)

    timestamp = datetime.utcnow()
    return json_data, player_stats, timestamp, args.player_name[0]

async def main(pkl_data=None):
    # because of streamlit "magic" functions, docstrings get printed in the dashboard:
    # pylint be upset but this is me showing where the docstring would go:

    # """
    # Main Streamlit function. It generates the sidebar, preps the data for viewing,
    # and displays the correct page based on the user's sidebar selection.
    # """

    if not pkl_data:
        json_data, player_stats, timestamp, player_name = parse_cli()
    else:
        # if we have a pkl file, we unpack that instead of parsing the CLI
        json_data = pkl_data.player_stats[0]
        player_name = pkl_data.player_name[0]
        data = jsonpickle.decode(json_data)
        player_stats = PlayerStat(int(pkl_data.player_id[0]), data)
        timestamp = datetime.utcnow()

    # Set page title and favicon.
    st.set_page_config(
        page_title=f"{player_name}'s POG Statistics",
        page_icon=POG_FAVICON
    )
    activate_css()
    st.image(POG_BANNER)

    st.sidebar.image(POG_LOGO, width=150)
    st.sidebar.title("Page Selector")
    app_mode = st.sidebar.selectbox(
        "Mostly for dev use",
        ["Display dashboard", "Raw player data", "Streamlit source code"]
        )
    if app_mode == "Display dashboard":
        st.sidebar.success(f"Data from: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        uploaded_file = st.sidebar.file_uploader(
            "Do you have a player data file?\nUpload it here:",
            type=["pkl", "json"],
            help="You can generate player data from the Raw Player Data tab to upload here"
            )
        if uploaded_file is not None:
            # if a custom player pkl file is provided, we unpack it in the same way as we did before
            custom_pkl_file = pickle.load(uploaded_file)
            custom_unfrozen_data = jsonpickle.decode(custom_pkl_file.player_stats[0])
            custom_player_stats = PlayerStat(
                int(custom_pkl_file.player_id[0]),
                custom_unfrozen_data
                )
            display_dashboard(custom_player_stats, player_name)
        else:
            display_dashboard(player_stats, player_name)
    elif app_mode == "Raw player data":
        st.sidebar.success(f"Data from: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        display_json_stats(json_data)
    elif app_mode == "Streamlit source code":
        display_source_code()

def display_json_stats(json_data):
    """
    Streamlit layout for the dev/backend page with the player's stats as JSON object.
    """
    st.header("Want to build your own graphics?")
    st.subheader("Here's your player stats object:")
    st.markdown(json.dumps(json_data, sort_keys=True, indent=4))

def display_source_code():
    """
    Streamlit layout for the source code page with the interactive_stats.py displayed.
    """
    # query from github to allow us to more easily move
    # interactive_stats.py to a different ec2 instance later
    url = "http://bit.do/github-repo-shortened-link-because-apparently-we-need-a-perfect-lint-score"
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    st.code(text)

def display_dashboard(player_stats, player_name):
    # the only reason we have to pass player name around this way is because pylint gets upset
    # about too many instance attributes. Really the 10/10 score should not be forced like this.
    """
    Streamlit layout for the dashboard/home page with the player's stats.
    """
    st.title(f"{player_name}'s POG Statistics")
    st.caption(f"Discord id: {player_stats.player_id}")

    # I intentionally acronyms as subheaders because it's much more common in our community
    # KPM = kills per minute, KDR = kill/death ratio
    # I'm not a fan of writing HTML in python but there seems to be no pythonic way to customize
    # streamlit styling we have to use spans to ensure that text is printed across the same line
    st.markdown(
        f"<p class='h1'>Matches played: <span class='bold'>\
            {player_stats.num_matches_played}</span></p>",
        unsafe_allow_html=True
        )
    st.markdown(
        f"<p class='h1'>Time played: <span class='bold'>\
            {timedelta(minutes=player_stats.time_played)}",
        unsafe_allow_html=True
        )
    st.markdown(
        f"<p class='h1'>KDR: <span class='bold {get_color(player_stats.kdr)}'>\
            {round(player_stats.kdr, 2)}</span> | \
        KPM: <span class='bold {get_color(player_stats.kpm)}'>\
            {round(player_stats.kpm, 2)}</span></p>",
        unsafe_allow_html=True
        )
    st.markdown(
        f"<p class='h1'>Percentage of matches as captain: <span class='bold'>\
            {round(player_stats.cpm*100, 2)}%</span></p>",
        unsafe_allow_html=True
        )
    st.markdown(
        f"<p class='h1'>Average kills per match: <span class='bold'>\
            {round(player_stats.kills_per_match, 2)}</span></p>",
        unsafe_allow_html=True
        )

    # pylint didn't want any more local variables so now we have some code that looks
    # more confusing than it needs to be:
    st.header("Total Score per Loadout")
    st.bar_chart(pd.DataFrame.from_dict(
        player_stats.loadout_scores,
        orient='index',
        columns=['Score']
        ))

    df_loadout_kdrs = pd.DataFrame.from_dict(
        player_stats.loadout_kdrs,
        orient='index',
        columns=['KDR']
        )
    st.header("KDR per Loadout")
    st.bar_chart(df_loadout_kdrs)

    win_loss_fig, axis = plt.subplots()
    axis.pie(
        [player_stats.matches_won, player_stats.matches_lost],
        labels=["Wins", "Losses"],
        autopct='%1.1f%%',
        startangle=90
        )
    axis.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(win_loss_fig)

    # we use a numpy matrix of 1s and 0s to create a visualization of games played vs not
    # played, where a colored square of the matrix indicates a match played a gray square
    # indicates not played. It's a bit like a waffle chart, but simpler since it doesn't
    # need to group values
    matches_played_matrix = np.zeros(max(player_stats.matches) + 1)

    for match in player_stats.matches:  # one-hot encoding matches played into an array of zeros
        matches_played_matrix[match] = 1

    st.header("Matches Played vs Not Played")
    st.markdown("Last matches from most recent to least recent, where green = match played")
    # because we want to display a square matrix, we drop values at the start of the array.
    # We can use a 32x32 matrix by default, but ask the user. This is fine because we don't
    # have match data before 569 anyways, and the graph isn't designed to be perfect,
    # just a quick visual representation of matches played vs not played recently

    grid_size = st.slider(
        'How many matches to show?',
        min_value=1,
        max_value=get_max_grid_size(matches_played_matrix),
        value=32
        )
    # select only the most recent n*n matches, so we can display a square matrix.
    # Here we choose 32*32 = 1024 by default.
    matches_played_matrix = matches_played_matrix[-(grid_size*grid_size):]
    matches_played_matrix = np.reshape(matches_played_matrix, (grid_size, grid_size))

    # Define colormap for matches played vs not played
    cmapmine = ListedColormap(['grey', 'lawngreen'], N=2)

    # Plot the matrix
    matches_played_fig = plt.figure()
    plt.imshow(matches_played_matrix, cmap=cmapmine, vmin=0, vmax=1)
    matches_played_fig.axes[0].set_xticks([]) # remove x axis labels
    matches_played_fig.axes[0].set_yticks([]) # remove y axis labels
    st.pyplot(matches_played_fig)

    player_stats.pick_order.values()

    st.header("Pick order")

    # we can't use player_stats.pick_order.keys() to get the pick order because it
    # sometimes has values outside 2-6. Values of 0 and 1 in the keys are for captain,
    # and values above 12 occur if there are substitutions mid-match. To make this graph
    # cleaner, we'll only include the first 10 picks, and start the index at 2 to ignore
    # the captain slots 0 and 1.

    # I had to remove this line below to keep lint happy... Situations like these though is why
    # PEP 20 exists and a 10/10 lint score isn't always the best thing:
        # pick_order_index = range(2, 12)

    y_pos = np.arange(len(range(2, 12)))
    pick_order = []
    for label in range(2, 12):
        if str(label) in player_stats.pick_order.keys():
            pick_order.append(player_stats.pick_order[str(label)])
        else:
            # if a player was never picked in a given slot, set number of picks for that slot to 0
            pick_order.append(0)

    # Plot the horizontal bar chart
    plt.rcdefaults()
    pick_order_fig, axes = plt.subplots()
    axes.barh(y_pos, pick_order, align='center')

    # For people viewing the graph, we are going to start the labels at 1 which means first pick
    # this is much more intuitive for an end user who doesn't know that 0 and 1 are for captains
    axes.set_yticks(y_pos, labels=range(1, 11))
    axes.invert_yaxis()
    axes.set_xlabel('Number of times picked')
    st.pyplot(pick_order_fig)

if __name__ == "__main__":
    # we run this asynchronously to avoid blocking the main thread when starting streamlit.
    # Also, if we ever want to integrate class methods from stats.py,
    # we'll need to await the function calls.
    asyncio.run(main(pkl_data=fetch_pkl())) # delete pkl_data argument for production
    #asyncio.run(main()) # delete pkl_data argument for production

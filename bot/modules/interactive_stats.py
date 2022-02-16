""" 
STREAMLIT APP FOR PLANETSIDE OPEN GAMES
Development can be followed at https://github.com/monkeydg/POG-bot/tree/stats
This is a web app that displays player and game statistics for Planetside Open Games.
Planetside Open Games, or POG, is a 6v6 matchmaking bot for the video game Planetside 2,
built around discord, with integrations into the game's API, Mongodb, AWS, and Teamspeak 3.
"""

# The goal is to avoid imports from the rest of the project where possible, so that we can
# later deploy the streamlit module on a separate ec2 instance with minimal refactoring
import os
import argparse
import asyncio
import json
import pickle
import urllib
from datetime import datetime, timedelta
from logging import getLogger
from collections import Counter
import jsonpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.figure_factory as ff
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
POG_LOGO = "https://media.discordapp.net/attachments/739231714554937455/739522071423614996/logo_png.png"
POG_BANNER = "https://cdn.discordapp.com/attachments/786308579015655474/942073850861072414/Banner_Planetside_Open_Games.png"

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

class PlayerStat:
    """
    Class used to track a player's statistics in a pythonic way instead of a dictionary.
    The properties of this class are calculated from the data in the dictionary used to
    initialize an instance of the class.
    """
    def __init__(self, player_id, name, data):
        self.player_id = player_id
        self.name = name
        self.matches = data["matches"]
        self.matches_won = data["matches_won"]
        self.matches_lost = data["matches_lost"]
        self.time_played = data["time_played"]
        self.times_captain = data["times_captain"]
        self.pick_order = data["pick_order"]
        self.loadouts = {}
        for loadout_data in data["loadouts"].values():
            loadout_id = loadout_data["id"]
            self.loadouts[loadout_id] = LoadoutStats(loadout_id, loadout_data)
        self.loadout_scores = self.generate_loadout_scores_dict()
        self.loadout_kdrs = self.generate_loadout_kdrs_dict()

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

class MatchlogStat:
    """
    Class used to track a match's statistics in a pythonic way instead of a dictionary.
    The properties of this class are calculated from the data in the dictionary used to
    initialize an instance of the class.
    """
    class CaptainStat:
        def __init__(self, captain_data):
            self.captain_id = captain_data["player"]
            self.captain_team = captain_data["team"]
            self.captain_timestamp = datetime.fromtimestamp(captain_data["timestamp"])

    class FactionStat:
        def __init__(self, faction_data):
            self.faction_name = faction_data["faction"]
            self.faction_team = faction_data["team"]
            self.faction_timestamp = datetime.fromtimestamp(faction_data["timestamp"])

    def __init__(self, match_data):
        self.match_id = match_data["_id"]
        self.match_start = datetime.fromtimestamp(match_data["match_launching"])
        self.match_end = datetime.fromtimestamp(match_data["match_over"])
        self.teams_end = datetime.fromtimestamp(match_data["teams_done"])
        self.rounds_start = [datetime.fromtimestamp(match_data["rounds"][0]["timestamp"]), datetime.fromtimestamp(match_data["rounds"][2]["timestamp"])]
        self.rounds_end = [datetime.fromtimestamp(match_data["rounds"][1]["timestamp"]), datetime.fromtimestamp(match_data["rounds"][3]["timestamp"])]
        self.captains = [self.CaptainStat(match_data["captains"][0]), self.CaptainStat(match_data["captains"][1])]
        self.factions = [self.FactionStat(match_data["factions"][0]), self.FactionStat(match_data["factions"][1])]

    @property
    def switching_sides_wait(self):
        """ Returns the time spent between rounds to switch sides """
        return self.rounds_start[1] - self.rounds_end[0]

    @property
    def login_wait(self):
        """ Returns the time spent between faction selected and round 1 beginning"""
        return self.rounds_start[0] - self.factions[1].faction_timestamp

    @property
    def faction_wait(self):
        """ Returns the time spent waiting for faction selection"""
        return self.factions[1].faction_timestamp - self.teams_end

    @property
    def team_pick_wait(self):
        """ Returns the time spent waiting for team selection"""
        return self.teams_end - self.captains[1].captain_timestamp

    @property
    def captain_pick_wait(self):
        """ Returns the time spent waiting for captain selection"""
        return self.captains[1].captain_timestamp - self.match_start

    @property
    def total_wait(self):
        """ Returns the time spent between match start and end where players
        were not playing (eg picking teams, switching sides)
        """
        return sum([self.login_wait, self.faction_wait, self.team_pick_wait, self.captain_pick_wait, self.switching_sides_wait])

def get_all_captains(all_matches):
    """ Returns a counted dict of all captains in the matchlog """
    captains_list = []
    for match in all_matches:
        for captain in match.captains:
            captains_list.append(captain.captain_id)
    return Counter(captains_list).most_common()

def get_teams_factions(all_matches):
    """ Returns two counted dicts of factions selected in the matchlog (one dict per team) """
    team_0_factions = []
    team_1_factions = []
    for match in all_matches:
        team_0_factions.append(match.faction[0].faction_name)
        team_1_factions.append(match.faction[1].faction_name)
    return Counter(team_0_factions).most_common(), Counter(team_1_factions).most_common()

def get_match_start_hours(all_matches):
    """ Returns a counted dict of what hour matches start in the matchlog """
    hours_list = []
    for match in all_matches:
        hours_list.append(match.match_start.hour)
    return sorted(Counter(hours_list).items())

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

def dump_pkl(json_file, file_path="cli_args.pkl"):
    """
    This is only used for dumping a json file for standalone testing/debugging
    """
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(json_file, pkl_file)

def fetch_pkl(file_path="cli_args.pkl"):
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
    parser.add_argument(
        '--matchlog_stats',
        action='append',
        default=[],
        help="JSON representation of a MatchlogStat object containing match data about POG"
        )

    try:
        args = parser.parse_args() # get player id, name, player stats, and match stats from CLI arguments

        # initiate a new PlayerStats object with the input from CLI arguments
        player_data_frozen = args.player_stats[0]
        player_data_unfrozen = jsonpickle.decode(player_data_frozen)
        player_stats = PlayerStat(int(args.player_id[0]), args.player_name[0], player_data_unfrozen)

        # initiate a new MatchStats object with the input from CLI arguments
        matchlog_data_frozen = args.matchlog_stats[0]
        matchlog_data_unfrozen = jsonpickle.decode(matchlog_data_frozen)
        all_matchlog_stats = [MatchlogStat(match) for match in matchlog_data_unfrozen]
    except SystemExit as exception:
        # This exception will be raised if an invalid command line argument is used.
        # Streamlit doesn't exit gracefully so we have to force it to stop ourselves.
        # This exists streamlit subprocess without disrupting the main discord bot process.
        log.warning("Invalid command line argument. Exiting Streamlit app.")
        os._exit(exception.code)

    timestamp = datetime.utcnow()
    return player_data_frozen, player_stats, matchlog_data_frozen, all_matchlog_stats, timestamp

async def main(pkl_data=None):
    if not pkl_data:
        player_data_frozen, player_stats, matchlog_stats, timestamp = parse_cli()
    else:
        args = pkl_data
        player_data_frozen = args.player_stats[0]
        player_data_unfrozen = jsonpickle.decode(player_data_frozen)
        player_stats = PlayerStat(int(args.player_id[0]), args.player_name[0], player_data_unfrozen)

        # initiate a new MatchStats object with the input from CLI arguments
        matchlog_data_frozen = args.matchlog_stats[0]
        matchlog_data_unfrozen = jsonpickle.decode(matchlog_data_frozen)
        all_matchlog_stats = [MatchlogStat(match) for match in matchlog_data_unfrozen]

    # Set page title and favicon.
    st.set_page_config(
        page_title=f"{player_stats.name}'s POG Statistics",
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
    st.sidebar.success(f"Data from: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if app_mode == "Display dashboard":
        display_dashboard(player_stats)
    elif app_mode == "Match logs":
        display_dashboard(all_matchlog_stats)
    elif app_mode == "Raw player data":
        display_json_stats(player_data_frozen)
    elif app_mode == "Streamlit source code":
        display_source_code()

def display_matchlogs(all_matchlog_stats):
    """
    Streamlit layout for the match logs over time page.
    """
    st.title(f"Match Logs Statistics")
    st.caption(f"Match data collected from match 569 onwards")

    st.markdown(
        f"<p class='h1'>Total matches played: <span class='bold'>\
            {len(all_matchlog_stats)}</span></p>",
        unsafe_allow_html=True
        )
    st.markdown(
        f"<p class='h1'>Total time played: <span class='bold'>\
            {timedelta(minutes=len(all_matchlog_stats)*MATCH_LENGTH)}",
        unsafe_allow_html=True
        )
    
    import streamlit as st
    import numpy as np

    # Add histogram data
    x1 = sum([match.login_wait for match in all_matchlog_stats])
    x2 = sum([match.faction_wait for match in all_matchlog_stats])
    x3 = sum([match.team_pick_wait for match in all_matchlog_stats])

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ['Group 1', 'Group 2', 'Group 3']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)


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
    url = "https://raw.githubusercontent.com/monkeydg/POG-bot/20c6aba563d1ef3955d9ab0b269c70f34d7eda3a/bot/modules/interactive_stats.py"
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    st.code(text)

def display_dashboard(player_stats):
    """
    Streamlit layout for the dashboard/home page with the player's stats.
    """
    st.title(f"{player_stats.name}'s POG Statistics")
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
            {round(player_stats.cpm, 2)*100}%</span></p>",
        unsafe_allow_html=True
        )
    st.markdown(
        f"<p class='h1'>Average kills per match: <span class='bold'>\
            {round(player_stats.kills_per_match, 2)}</span></p>",
        unsafe_allow_html=True
        )

    df_loadout_scores = pd.DataFrame.from_dict(
        player_stats.loadout_scores,
        orient='index',
        columns=['Score']
        )
    st.header("Total Score per Loadout")
    st.bar_chart(df_loadout_scores)

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
    st.markdown("Last 1024 matches, where green = match played")
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
    pick_order_index = range(2, 12)

    y_pos = np.arange(len(pick_order_index))
    pick_order = []
    for label in pick_order_index:
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

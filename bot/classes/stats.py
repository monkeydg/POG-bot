import modules.database as db
import modules.config as cfg
import modules.tools as tools
import operator
import subprocess
import multiprocessing as mp
import os
import signal
import jsonpickle
from interactive_stats import dump_pkl
from datetime import datetime
import operator
from functools import reduce


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
        self.matches_won = data["match_stats"]["nb_won"]
        self.matches_lost = data["match_stats"]["nb_lost"]
        self.time_played = data["time_played"]
        self.times_captain = data["times_captain"]
        self.pick_order = tools.AutoDict(data["pick_order"])
        self.loadouts = {}
        for loadout_data in data["loadouts"]:
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

    @classmethod
    async def get_from_database(cls, player_id, name):
        dta = await db.async_db_call(db.get_element, "player_stats", player_id)
        return cls(player_id, name=name, data=dta)

    def add_data(self, match_id: int, time_played, player_score):
        self.matches.append(match_id)
        if player_score.team.won_match:
            self.matches_won += 1
        else:
            self.matches_lost += 1
        self.time_played += time_played
        self.times_captain += int(player_score.is_captain)
        self.pick_order.auto_add(str(player_score.pick_index), 1)
        for l_id in player_score.loadouts.keys():
            loadout = player_score.loadouts[l_id]
            if l_id in self.loadouts:
                self.loadouts[l_id].add_data(loadout)
            else:
                self.loadouts[l_id] = LoadoutStats(l_id, loadout.get_data())

    def get_data(self):
        dta = dict()
        dta["_id"] = self.id
        dta["matches"] = self.matches
        dta["match_stats"] = {
            "nb_won": self.matches_won,
            "nb_lost": self.matches_lost,
        }
        dta["time_played"] = self.time_played
        dta["times_captain"] = self.times_captain
        dta["pick_order"] = self.pick_order
        dta["loadouts"] = [loadout.get_data() for loadout in self.loadouts.values()]
        return dta

class LoadoutStats:
    def __init__(self, l_id, data=None):
        self.id = l_id
        if data:
            self.weight = data["weight"]
            self.kills = data["kills"]
            self.deaths = data["deaths"]
            self.net = data["net"]
            self.score = data["score"]
        else:
            self.weight = 0
            self.kills = 0
            self.deaths = 0
            self.net = 0
            self.score = 0

    def add_data(self, loadout):
        self.weight += loadout.weight
        self.kills += loadout.kills
        self.deaths += loadout.deaths
        self.net += loadout.net
        self.score += loadout.score

    def get_data(self):
        data = {"id": self.id,
                "score": self.score,
                "net": self.net,
                "deaths": self.deaths,
                "kills": self.kills,
                "weight": self.weight,
                }
        return data

class MatchlogStat:
    """
    Class used to track a match's statistics in a pythonic way instead of a dictionary.
    The properties of this class are calculated from the data in the dictionary used to
    initialize an instance of the class.
    """
    _all_matchlogs = {}

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
        MatchlogStat._all_matchlogs[self.match_id] = self
    
    @classmethod
    def get_all(self):
        return self._all_matchlogs
    
    @classmethod
    def delete_one(self, match_data):
        print(self._all_matchlogs)
        id_to_delete = match_data["_id"]
        print(id_to_delete)
        del (self._all_matchlogs[id_to_delete])

    @property
    def switching_sides_wait(self):
        """ Returns the time spent between rounds to switch sides """
        # the operator module allows us to add datetimes while maintaining it as a timedelta object
        return reduce(operator.sub, self.rounds_start[1] - self.rounds_end[0])

    @property
    def login_wait(self):
        """ Returns the time spent between faction selected and round 1 beginning"""
        return reduce(operator.sub, self.rounds_start[0] - self.factions[1].faction_timestamp)

    @property
    def faction_wait(self):
        """ Returns the time spent waiting for faction selection"""
        return reduce(operator.sub, self.factions[1].faction_timestamp - self.teams_end)

    @property
    def team_pick_wait(self):
        """ Returns the time spent waiting for team selection"""
        return reduce(operator.sub, self.teams_end - self.captains[1].captain_timestamp)

    @property
    def captain_pick_wait(self):
        """ Returns the time spent waiting for captain selection"""
        return reduce(operator.sub, self.captains[1].captain_timestamp - self.match_start)

    @property
    def total_wait(self):
        """ Returns the time spent between match start and end where players
        were not playing (eg picking teams, switching sides)
        """
        return reduce(operator.add, self.login_wait, self.faction_wait, self.team_pick_wait, self.captain_pick_wait, self.switching_sides_wait)


class StreamlitApp:
    _counter = 0 # we need to keep track of how many streamlit instances we spawn so that we don't overload our server with separate streamlit instances running on hundreds of ports
    def __init__(self, player_id, player_name, player_stats, matchlog_stats):
        StreamlitApp._counter += 1
        self.id = StreamlitApp._counter
        self.requestor = player_id

        
        self.player_id = player_id
        self.player_name = player_name
        self.player_stats = player_stats
        self.matchlog_stats = matchlog_stats

        #dump_pkl(matchlog_stats, "matchlog_stats.pkl")

        self.server_port = 8500 + StreamlitApp._counter  # we start at server port 8501, the streamlit default, then increment by one for each streamlit instance spawned
        # if there is already a streamlit instance running for the player that requested it, we don't need to spawn another one.
        #todo: add code here ^^

        # we need to pass the player id and player data to the streamlit app before running it
        # choosing to use command line arguments is much better than URL query strings because it makes it impossible for players to access other players' streamlit pages
        # if URL query strings end up being desired, read https://github.com/streamlit/streamlit/issues/430
        # passing the data as a json object is not very pythonic BUT it is the most flexible if we move the streamlit app to a different environment. I've decided to use a json object for now just for this added flexibility.
        # using a json file also has the added benefit of not needing to do any database opeterations within our streamlit app, making it much more modular. Also jsonpickle makes it easy to change between python objects and json

        self.process = subprocess.Popen(
            ["streamlit", "run", "interactive_stats.py",
            "--server.port", str(self.server_port),
            "--",
            "--player_id", str(self.player_id),
            "--player_name", str(self.player_name),
            "--player_stats", jsonpickle.encode(self.player_stats)
            ])
            # broken below due to too many args in CLI. This was janky af anyways for match data since it's so big, so I'm going to build this better:
            #"--matchlog_stats", jsonpickle.encode(self.matchlog_stats)]) #, preexec_fn=os.setsid)  # unfortunately we can't use setsid on windows, only linux.
        self.url = f"http://localhost:{self.server_port}/"

    def kill(self):  # unfortunately we can't use this method on windows, only linux. TLDR don't use this method on windows lol.
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        return "test"

class MyFancyClass(object):
    
    def __init__(self, name):
        self.name = name
    
    def do_something(self):
        proc_name = mp.current_process().name
        print ('Doing something fancy in %s for %s!' % (proc_name, self.name))
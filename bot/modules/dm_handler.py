from classes import Player, PlayerStat
from match.classes import Match
from modules import streamlit
import modules.config as cfg
from display import AllStrings as disp, ContextWrapper
from logging import getLogger
import modules.stat_processor as stat_processor
import modules.spam_checker as spam_checker
import modules.streamlit as streamlit

log = getLogger("pog_bot")


async def on_dm(message):
    # Check if too many requests from this user:
    if await spam_checker.is_spam(message.author, message.channel):
        return
    if message.content[:1] == "=":
        message.content = message.content[1:]
    if message.content.lower().startswith(("stat", "stats", "s")):
        await on_stats(message.author)
    elif message.content.lower().startswith(("modmail ", "dm ", "staff ")):
        i = message.content.index(' ')
        message.content = message.content[i+1:]
        player = Player.get(message.author.id)
        await disp.BOT_DM.send(ContextWrapper.channel(cfg.channels["staff"]), player=player, msg=message)
        await disp.BOT_DM_RECEIVED.send(message.author)
    elif message.content.lower().startswith(("help", "h")):
        await disp.HELP.send(message.author, is_dm=True)
    spam_checker.unlock(message.author.id)


async def on_stats(user):
    player = Player.get(user.id)
    if not player:
        await disp.NO_RULE.send(user, "stats", cfg.channels["rules"]) # if the player isn't registered, he can't see his stats. We send them to the rules channel.
        return
    log.info(f"Stats request from player id: [{player.id}], name: [{player.name}]")
    all_stats = await PlayerStat.get_from_database(player.id, player.name)
    recent_stats = await stat_processor.get_new_stats(Match, all_stats)
    streamlit_url = await streamlit.generate_streamlit_url(all_stats, pog_stats) # calls a method from streamlit module that generates a streamlit url after passing in the player stats from the playerStats collection in mongodb and newMatches 
    await disp.DISPLAY_STATS.send(user, stats=all_stats, recent_stats=recent_stats, streamlit_url=streamlit_url) # displays streamlit url in discord bot DM using the strings.py and embeds.py modules


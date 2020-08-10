"""main.py

Initialize everything, attach the general handlers, run the client.
The application should be launched from this file
"""

# discord.py
from discord.ext import commands
from discord.ext.commands import Bot
from discord import Status, DMChannel

# Other modules
from asyncio import sleep
from random import seed
from datetime import datetime as dt

# Custom modules
import modules.config as cfg
from modules.display import send, channelSend, edit
from modules.display import init as displayInit
from modules.spam import isSpam, unlock
from modules.exceptions import ElementNotFound
from modules.database import init as dbInit, getAllPlayers, getAllMaps
from modules.enumerations import PlayerStatus
from modules.tools import isAdmin
from modules.loader import init as cogInit, isAllLocked, unlockAll

# Modules for the custom classes
from matches import onPlayerInactive, onPlayerActive, init as matchesInit
from classes.players import Player, getPlayer
from classes.accounts import AccountHander


def _addMainHandlers(client):
    """_addMainHandlers, private function
        Parameters
        ----------
        client : discord.py bot
            Our bot object
    """

    rulesMsg = None # Will contain message object representing the rules message, global variable

    # help command, works in all channels
    @client.command(aliases=['h'])
    @commands.guild_only()
    async def help(ctx):
        await send("HELP",ctx)

    # Slight anti-spam: prevent the user to input a command if the last one isn't yet processed
    # Useful for the long processes like ps2 api, database or spreadsheet calls
    @client.event
    async def on_message(message):
        if message.author == client.user: # if bot, do nothing
            await client.process_commands(message)
            return
        if isinstance(message.channel, DMChannel): # if dm, print in console and ignore the message
            print(message.author.name + ": " +message.content)
            return
        if message.channel.id not in (cfg.discord_ids["lobby"], cfg.discord_ids["register"], *cfg.discord_ids["matches"]):
            return
        if isAllLocked():
            if not isAdmin(message.author):
                return
            # Admins can still use bot when locked
        if await isSpam(message):
            return
        await client.process_commands(message) # if not spam, process
        await sleep(0.5)
        unlock(message.author.id) # call finished, we can release user

    # on ready
    @client.event
    async def on_ready():
        print('Client is online')

        # fetch rule message, remove all reaction but the bot's
        global rulesMsg
        rulesMsg = await client.get_channel(cfg.discord_ids["rules"]).fetch_message(cfg.discord_ids["rules_msg"])
        await rulesMsg.clear_reactions()
        await sleep(0.2)
        await rulesMsg.add_reaction('✅')

    # Global command error handler
    @client.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandNotFound): # Unknown command
            if isAllLocked():
                await send("BOT_IS_LOCKED", ctx)
                return
            await send("INVALID_COMMAND",ctx)
            return
        if isinstance(error, commands.errors.CheckFailure): # Unauthorized command
            cogName = ctx.command.cog.qualified_name
            if cogName == "admin":
                await send("NO_PERMISSION", ctx, ctx.command.name)
                return
            try:
                channelId = cfg.discord_ids[cogName]
                channelStr = ""
                if isinstance(channelId, list):
                    channelStr = " channels " + ", ".join(f'<#{id}>' for id in channelId)
                else:
                    channelStr = f' channel <#{channelId}>'
                await send("WRONG_CHANNEL", ctx, ctx.command.name, channelStr) # Send the use back to the right channel
            except KeyError: # Should not happen
                await send("UNKNOWN_ERROR", ctx, "Channel key error")
            return
        # These are annoying error generated by discord.py when user input quotes (")
        if isinstance(error, commands.errors.InvalidEndOfQuotedStringError) or isinstance(error, commands.errors.ExpectedClosingQuoteError) or isinstance(error, commands.errors.UnexpectedQuoteError):
            await send("INVALID_STR", ctx, '"') # Tell the user not to use quotes
            return
        await send("UNKNOWN_ERROR", ctx, type(error.original).__name__) # Print unhandled error
        raise error

    # Reaction update handler (for rule acceptance)
    @client.event
    async def on_raw_reaction_add(payload): # Has to be on_raw cause the message already exists when the bot starts
        if payload.member == None or payload.member.bot: # If bot, do nothing
            return
        if isAllLocked():
            if not isAdmin(payload.member):
                return
        if payload.message_id == cfg.discord_ids["rules_msg"]: # reaction to the rule message?
            global rulesMsg
            if str(payload.emoji) == "✅":
                try:
                    getPlayer(payload.member.id)
                except ElementNotFound: # if new player
                    Player(payload.member.name, payload.member.id) # create a new profile
                    await channelSend("REG_RULES", cfg.discord_ids["register"], payload.member.mention) # they can now register
                    registered = payload.member.guild.get_role(cfg.discord_ids["registered_role"])
                    info = payload.member.guild.get_role(cfg.discord_ids["info_role"])
                    await payload.member.add_roles(registered)
                    await payload.member.remove_roles(info)

            await rulesMsg.remove_reaction(payload.emoji, payload.member) # In any case remove the reaction, message is to stay clean

    # Reaction update handler (for accounts)
    @client.event
    async def on_reaction_add(reaction, user):
        try:
            player = getPlayer(user.id)
        except ElementNotFound:
            return
        if player.hasOwnAccount:
            return
        if player.status != PlayerStatus.IS_PLAYING:
            return
        if player.active.account == None:
            return
        # If we reach this point, we know player has been given an account
        account = player.active.account
        if reaction.message.id != account.message.id: # chack if it's the right message
            return
        if account.isValidated: # Check if user didn't already react
            return
        if str(reaction.emoji) == "✅": # If everything is fine, account is validated
            account.validate()
            await edit("ACC_UPDATE", account.message, account=account)
            await account.message.remove_reaction(reaction.emoji,client.user)

    # Status update handler (for inactivity)
    @client.event
    async def on_member_update(before, after):
        try:
            player = getPlayer(after.id)
        except ElementNotFound:
            return
        if after.status == Status.offline:
            onPlayerInactive(player)
        else:
            onPlayerActive(player)

# TODO: testing, to be removed
def _test(client):
    from test2 import testHand
    testHand(client)



def main(launchStr=""):
    # Init order MATTERS

    # Seeding random generator
    seed(dt.now())

    # Get data from the config file
    cfg.getConfig(f"config{launchStr}.cfg")

    # Set up command prefix
    client = commands.Bot(command_prefix=cfg.general["command_prefix"])

    # Remove default help
    client.remove_command('help')

    # Initialise db and get all t=xhe registered users and all maps from it
    dbInit(cfg.database)
    getAllPlayers()
    getAllMaps()

    # Get Account sheet from drive
    AccountHander.init(f"client_secret{launchStr}.json")

    # Initialise matches channels
    matchesInit(cfg.discord_ids["matches"])

    # Initialise display module
    displayInit(client)

    # Add main handlers
    _addMainHandlers(client)
    if launchStr=="_test":
        _test(client)

    # Add all cogs
    cogInit(client)
    unlockAll(client)

    # Run server
    client.run(cfg.general["token"])


if __name__ == "__main__":
    # execute only if run as a script
    # Use main() for production
    #main("_test")
    main()
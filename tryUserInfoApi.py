import logging
import os
# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# WebClient instantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.
client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
logger = logging.getLogger(__name__)

# ID of user you watch to fetch information for
user_id = "U08BG9BT5DF"

try:
    # Call the users.info method using the WebClient
    result = client.users_info(
        user=user_id
    )
    logger.info(result)
    print(result)

except SlackApiError as e:
    logger.error("Error fetching conversations: {}".format(e))
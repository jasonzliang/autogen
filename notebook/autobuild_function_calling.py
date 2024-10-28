#!/usr/bin/env python
# coding: utf-8

# # AutoBuild Agents function calling
# By: [Krishna Shedbalkar](https://github.com/krishnashed/)
# 
# In this notebook, we introduce a way for Agents created using `Autobuild` to do function calling. Developers can specify a function, function name and function description which will thereafter be assigned and executed by the most suitable agent created using AutoBuild.

# ## Requirement
# 
# AutoBuild require `pyautogen[autobuild]`, which can be installed by the following command:

# In[ ]:


get_ipython().run_line_magic('pip', 'install pyautogen[autobuild]')


# ## Step 1: Prepare configuration and some useful functions
# 
# Prepare a `config_file_or_env` for assistant agent to limit the choice of LLM you want to use in this task. This config can be a path of json file or a name of environment variable. A `default_llm_config` is also required for initialize the specific config of LLMs like seed, temperature, etc. Preventing UserProxy agent being called multiple times by adding `allow_repeat_speaker=agent_list[:-1]`

# In[1]:


import autogen
from autogen.agentchat.contrib.agent_builder import AgentBuilder

config_file_or_env = "OAI_CONFIG_LIST"
config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-4-1106-preview", "gpt-4"]})
llm_config = {
    "config_list": config_list,
    "timeout": 120,
}


def start_task(execution_task: str, agent_list: list):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], allow_repeat_speaker=agent_list[:-1], max_round=12)
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list})
    agent_list[0].initiate_chat(manager, message=execution_task)


# ## Step 2: Create a AgentBuilder
# 
# Create a `AgentBuilder` with the specified `config_path_or_env`. AgentBuilder will use `gpt-4` in default to complete the whole process, you can specify the `builder_model` and `agent_model` to other OpenAI model to match your task. You can also specify an open-source LLM supporting by vLLM and FastChat, see blog for more details.

# In[2]:


builder = AgentBuilder(
    config_file_or_env=config_file_or_env, builder_model="gpt-4-1106-preview", agent_model="gpt-4-1106-preview"
)


# ## Step 3: Specify a building task
# 
# Specify a building task with a general description. Building task will help build manager (a LLM) decide what agents should be built.

# In[10]:


building_task = "Analyze and list the trending topics in arxiv papers related to GPT-4"


# ## Step 4: Define functions
# 
# Define functions to be executed by the Agents of AutoBuild, further specify details like `name`, `description` and `function` of all the functions in an array called `list_of_functions` which will be passed to `builder.build()`

# In[12]:


import time
from datetime import datetime, timedelta
from typing import Dict

import feedparser


def get_arxiv_paper_from_a_week(search_topic: str) -> Dict:
    # arXiv API endpoint
    url = "http://export.arxiv.org/api/query?"

    # Search parameters
    max_results = 10

    query = (
        f"{url}search_query=all:{search_topic}&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending"
    )

    # Parse the feed
    feed = feedparser.parse(query)

    now = datetime.now()
    week_ago = now - timedelta(weeks=1)

    papers = []

    # Get papers from last week
    for entry in feed.entries:
        published_time = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
        if published_time > week_ago:
            list_of_authors = ", ".join(author.name for author in entry.authors)

            papers.append(
                {
                    "title": entry.title,
                    "authors": list_of_authors,
                    "published_on": time.strftime("%B %d, %Y", entry.published_parsed),
                    "summary": entry.summary,
                    "link": entry.link,
                }
            )

    return papers


list_of_functions = [
    {
        "name": "get_arxiv_paper_from_a_week",
        "description": "Get arxiv papers published in last week",
        "function": get_arxiv_paper_from_a_week,
    }
]


# ## Step 5: build group chat agents
# 
# Use `build()` to let build manager (the specified `builder_model`) complete the group chat agents generation. Specify `list_of_functions` to be used by the Agents

# In[13]:


agent_list, agent_configs = builder.build(building_task, llm_config, list_of_functions, max_agents=3)


# Here you can see that Function `exec_python` has been associated with `ArxivAPI_Expert` Agent.

# ## Step 6: execute task
# 
# Let agents generated in `build()` to complete the task collaboratively in a group chat.

# In[14]:


start_task(execution_task=building_task, agent_list=agent_list)


# ## Step 7 (Optional): clear all agents and prepare for the next task
# 
# You can clear all agents generated in this task by the following code if your task is completed or the next task is largely different from the current task. If the agent's backbone is an open-source LLM, this process will also shut down the endpoint server. If necessary, you can use `recycle_endpoint=False` to retain the previous open-source LLMs' endpoint server.

# In[15]:


builder.clear_all_agents(recycle_endpoint=True)


# ## Save & load configs
# 
# You can save all necessary information of the built group chat agents. Here is a case for those agents generated in the above task:

# In[16]:


saved_path = builder.save()


# In[ ]:





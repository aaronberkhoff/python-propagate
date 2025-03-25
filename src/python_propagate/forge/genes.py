from copy import copy, deepcopy
import numpy as np
from python_propagate.agents import Agent
from typing import Iterable
import itertools
# import re


def set_subattribute(agent, public_attrs, attribute, value, name_index = None):
    """
    Sets a subattribute of an agent object by iterating over its public attributes.
    
    Parameters:
        agent (object): The agent instance to modify.
        public_attrs (list): A list of attribute names (strings) of the agent that are considered public.
        attribute (str): The name of the subattribute to set.
        value (any): The value to assign to the subattribute.
        i (int): An index used to update the agent's name.
        
    Returns:
        int: The count of subattributes successfully set.
        
    Raises:
        AttributeError: If the subattribute is not found in any of the public attributes.
    """
    set_bool = False

    for att in public_attrs:
        try:
            items = getattr(agent, att)  # Attempt to get the attribute
        except (AttributeError, TypeError):
            continue

        try:
            for obj in items:
                if hasattr(obj, attribute):
                    setattr(obj, attribute, value)
                    # Update the agent name to reflect the change
                    if name_index is not None:
                        agent.name = f"{agent.name}__{attribute}{name_index}"
                    set_bool = True
        except (AttributeError, TypeError):
            # If 'items' is not iterable or doesn't support iteration, skip it
            continue

    if not set_bool:
        print(
            f'<{attribute}> is not a settable attribute of {agent.name}'
        )

    return agent

def get_subattribute(agent, public_attrs, attribute):
    """
    Retrieves a subattribute from an agent's public attributes.
    
    Parameters:
        agent (object): The agent instance to inspect.
        public_attrs (list): A list of attribute names (strings) that are considered public on the agent.
        attribute (str): The name of the subattribute to retrieve.
    
    Returns:
        list: A list of values for the subattribute found in the agent's public attributes.
        
    Raises:
        AttributeError: If the subattribute is not found in any of the public attributes.
    """

    for att in public_attrs:
        try:
            items = getattr(agent, att)  # Get the attribute from the agent.
        except (AttributeError, TypeError):
            continue

        try:
            # If items is iterable, loop through its elements.
            for obj in items:
                if hasattr(obj, attribute):
                    results = getattr(obj, attribute)
        except (AttributeError, TypeError):
            # If items isn't iterable, skip it.
            continue

    if results is None:
        raise AttributeError(
            f"'{attribute}' not found as a subattribute in any of the agent's public attributes"
        )

    return results


class Gene:

    def __init__(self, agent_base: Agent, attribute:str, samples:int, **kwargs):

        self.attribute = attribute
        self.agent_base = agent_base
        self.chromosomes = []
        self.samples = samples
        self.min = kwargs[attribute + '_min'] 
        self.max = kwargs[attribute + '_max'] 

        # values = np.array([self.min + i * step for i in range(samples)])
        values = np.linspace(self.min,self.max,samples)
        cnt = 0
        wcnt = 0
        # values = np.arange(self.min,self.max,(self.max - self.min)/samples)
        public_attrs = [attr for attr in dir(agent_base) if not attr.startswith("_")]

        for i, value in enumerate(values):
            agent = deepcopy(agent_base)
            set_bool = False  # Reset flag for each agent

            if not hasattr(agent_base, attribute):  # If the base agent doesn't have the attribute
                set_subattribute(agent=agent,
                                 public_attrs=public_attrs,
                                 attribute=attribute,
                                 value=value,
                                 name_index=i)
            else:
                setattr(agent, attribute, value)
                cnt += 1
                agent.name = f"{agent.name}__{attribute}{i}"

            self.chromosomes.append(agent)



        # self.chromosomes = np.array(chromosomes)
        
        pass

def combine(genes: Iterable):
    agents = []
    
    # Get all chromosomes across all genes
    chromosome_lists = [gene.chromosomes for gene in genes]
    public_attrs = [attr for attr in dir(genes[0].agent_base) if not attr.startswith("_")]

    # test = [chrom.manuevers[0].direction_ric for chrom in genes[0].chromosomes]
    # Iterate over every combination of chromosomes
    for chromosome_combination in itertools.product(*chromosome_lists):
        base_agent = deepcopy(chromosome_combination[0])  # Start with the first chromosome
        
        combined_suffixes = []  # Store suffixes for name construction
        
        for gene, chrom in zip(genes[1:], chromosome_combination[1:]):
            if hasattr(chrom,gene.attribute):
                setattr(base_agent, gene.attribute, getattr(chrom, gene.attribute))
                _, suffix = chrom.name.split("__")
                combined_suffixes.append(suffix)
            else:
                set_subattribute(base_agent,
                                 public_attrs,
                                 gene.attribute,
                                 get_subattribute(chrom, public_attrs, gene.attribute))
                _, suffix = chrom.name.split("__")
                combined_suffixes.append(suffix)
        
        # Construct the new name dynamically
        prefix, suffix1 = chromosome_combination[0].name.split("__")
        combined_name = f"{prefix}__{suffix1}_{'_'.join(combined_suffixes)}"
        base_agent.name = combined_name

        agents.append(base_agent)

    return agents

class Genes:
    def __init__(self,genes: Iterable[Gene]):
        self.genes = genes
        self.agents = combine(genes)
        pass

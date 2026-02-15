def federated_average(agents):

    global_actor = agents[0].actor.state_dict()

    for key in global_actor.keys():
        for i in range(1, len(agents)):
            global_actor[key] += agents[i].actor.state_dict()[key]
        global_actor[key] /= len(agents)

    return global_actor
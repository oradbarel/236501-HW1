import csv
from rosman_nimrod_maps import small, medium, large, segel


final = {}
final.update(small)
final.update(medium)
final.update(large)
final.update(segel)
test_envs = {}
for board_name, board in final.items():
    test_envs[board_name] = DragonBallEnv(board)

test_expanded = True
result = "results.csv" if test_expanded else "results_without_expanded.csv"
user_result = "my_results.csv"
BFS_agent = BFSAgent()
WAStar_agent = WeightedAStarAgent()
AStar_epsilon_agent = AStarEpsilonAgent()
weights = [0.1, 0.3, 0.5, 0.7, 0.9]
epsilons = weights.copy()
agents_search_function = [
    BFS_agent.search,
]

with open(user_result, 'w') as f:
  writer = csv.writer(f)
  for env_name, env in test_envs.items():
    data = [env_name]
    for agent in agents_search_function:
      actions, total_cost, expanded = agent(env)
      data += [total_cost, expanded, actions]
    for w in weights:
      actions, total_cost, expanded = WAStar_agent.search(env, w)
      if test_expanded:
         data += [total_cost, expanded, actions]
      else:
         data += [total_cost, actions]
    for espilon in epsilons:
      actions, total_cost, expanded = AStar_epsilon_agent.search(env, espilon)
      if test_expanded:
        data += [total_cost, expanded, actions]
      else:
        data += [total_cost, actions]


    writer.writerow(data)

def compare_csv(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        for row1, row2 in zip(reader1, reader2):
            if row1 != row2:
                return False
        return True
    
if compare_csv(result, user_result):
   print(f'Congrats, You have passed!')
else:
   print(f'Give it another try :)')

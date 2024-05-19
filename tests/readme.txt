Thank you for running the tests!

Key Points:
This assignment took some time, and now we need to focus on other courses. 
Unfortunately, we won't be available for further assistance. We really do apologize.

Test Overview:

The test scenarios are detailed in the "test_rosman_nimrod.py" file and include:

    1. Random small and medium maps.
    2. Specific edge cases.
    3. One large map to test robustness.
    4. Tests provided by the course stuff (segel tests).

Instructions:

    1. Spill out the contents of "benchmark.py" in the final cell of your Python notebook (replacing current benchmark).
    COPY THE CONTENT, DO NOT TRY TO USE benchmark.py
    2. Place "test_rosman_nimrod.py," "results.csv," and "results_without_expanded.csv" in the same folder as your Python notebook.
    3. Restart your Python notebook and execute it as usual.
    4. The test results will be printed out upon completion.
    5. Notice - the huge map is huge, really huge. The test can take 15 to 60 min (implementation dependent)

Additional Notes:

    Within "benchmark.py," there's a variable named "test_expanded" set to True by default.
    This setting tests the number of expanded nodes in WA* and A*-Epsilon algorithms.
    To disable it and only check Actions and Cost, set it to False.

Consider the following clarifications provided by Shadi:

    1. The Goal node (e.g., (63, True, True)) does NOT count as an expanded node.
    2. Nodes representing holes and G positions (e.g., (63, False, True)) do count as expanded nodes.
    3. Nodes representing holes and G positions can be improved and are treated like any other nodes,
       except successors are not created for them (we dot not enter the for loop checking their sons)
    4. Finding a better route for a node, is possible to any node your agent met, whether it is in the close_list or open_list(heapdict).
    5. HFocal function is the cost (g).
    6. Focal is managed by heapdict, similar to the open priority queue.
    The priority is determined by (g, position). While Shadi mentioned that the third tie-break is irrelevant, I think it is. 
    See map "Edge1" for an example.
    In the reception hour, Shadi recommended using heapdict for Focal and letting it decide on the third tie-break.

Thank you for your cooperation.
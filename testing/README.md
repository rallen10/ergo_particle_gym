Unit tests, integration tests, and system tests for the ergo_particle_gym library


## Smoke Testing

These tests consists of complete `train.py` commands that kick off very short training cycles on a variety of scenarios with a variety of inputs. The idea is that we just want to run the entire "system" for a very short amount of time to make sure nothing is broken right off the bat. Note that these test don't evaluate if the output is correct or expected; they simply test if all the "machinery" works when it's turned on.


+ Example: Run smoke tests across range of algorithms and scenarios to make sure everything kicks up correctly. By default the smoke tests are run in parallel on 8 cores, but you can choose the number of cores

    ```
    cd testing
    python smoke_tests.py --num-cores 12
    ```

+ Example: If any smoke tests fail when running in parallel, it can be hard to track down the exact error since all parallel threads output to the terminal in non-sequential order. Therefore, in the event of a smoke test failure it can be useful to re-run in serial to find the first that failed

    ```
    cd testing
    python smoke_tests.py --run-serial
    ```

+ Example: Run smoke tests that include old, out-date smoke tests that are ignored by default for brevity

	```
    cd testing
    python smoke_tests.py --run-archived
    ```

+ Example: Run subset of smoke tests that include the word "graph" in their input command
	```
    cd testing
    python smoke_tests.py --filter graph
    ```



# Unit & Integration Tests

Where smoke tests check that all of the pieces work together without breaking, the unit and integration make sure that individual components and small groups of components output correct/expected values. Note that due to sloppiness/laziness, the unit and integration tests are not separated out as they should be; they are all mixed together in the same files 

+ Example: Run all unit/integration tests
    ```
    cd testing
    nosetests --nologcapture --verbose --exe
    ```

+ Example: Run a specific unit/integration from a specific file
    ```
    cd testing
    nosetests test_mappo.py:TestCentralCriticNetwork2 --nologcapture --nocapture --verbose
    ```

+ Example: Run all unit/integration tests using multiprocessing to distrube tests across 10 cores. Note that you need the process-timeout flag because some tests take too long on the default
    ```
    cd testing
    nosetests --nologcapture --nocapture --verbose --exe --processes=12 --process-timeout=120
    ```

+ Note: some tests are fundamentally probabalistic; e.g. `mappo: central critic learning periodic function (this may take a while)`. They actually fail sometimes even if nothing is fundamentally wrong because they are just checking to see if the mean of some output is within some range. If you get a failure on one of these tests, just re-run it. If it continues to fail, something is probably wrong. (I know I should come up with a more rigorous/structured way to test probalistic outcomes, but it's not a priority right now)

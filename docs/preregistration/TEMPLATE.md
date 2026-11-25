# Pre-registration template

Copy this file to `docs/preregistration/<release>-<topic>.md` and commit it alone before any script or run that it covers. Replace the guidance under each heading. The release gate fails a changed pre-registration file that has a heading with the word Predictions and no heading with the word Power.

Status: written before the script exists and before any real call in this phase. Date: YYYY-MM-DD.
Subject: the package, version and files under test.

## Question

State the question in one or two sentences, including what it does not claim. Name the claimed behaviour and where the claim is written. State what was already known before this file was written, including any earlier measurement, saved evidence read, scratch computation (with its file hash) or probe call.

## Predictions

Give a numeric prediction for every statistic, with a tolerance, so that each one can hold or fail. Define each statistic exactly, including the interval method, the tail probability, the number of seeds or blocks and how seeds are derived. Give the decision threshold for each row. A prediction that fails stays in the record as written.

## Power of each null result

For every null result (a lower bound below a claim, no difference, no effect), state the smallest effect the check could find with 80 percent power, with the formula or simulation behind it and the run count it assumes. Say which events, inputs and pairs the statement covers and which it leaves out. If the check could not fail at its claim, say so and report the null as uninformative. Add a positive control where one is possible.

## Stopping rule

Fix the number of runs, seeds and rows before the first run. Say whether any replication is allowed, what triggers it and how both results are reported. State that nothing else is added after the first full run.

## Deviation log

Nothing above this heading may be edited after the commit that adds this file. Deviations are appended here with the date and the reason. Anything done after the pre-registration commit that was not in it goes here, including exploratory runs and test runs of the script.

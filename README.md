# -neural-network-development-index
hackathon case solution 


Description Data Recieved
The solutions are evaluated on two criteria: predicted future Index values and allocated energy from a newly discovered star

1) Index predictions are evaluated using RMSE metric

2) Energy allocation is also evaluated using RMSE metric and has a set of known factors that need to be taken into account.

Every galaxy has a certain limited potential for improvement in the index described by the following function:

Potential for increase in the Index = -np.log(Index+0.01)+3

Likely index increase dependent on potential for improvement and on extra energy availability is described by the following function:

Likely increase in the Index = extra energy * Potential for increase in the Index **2 / 1000

There are also several constraints:

in total there are 50000 zillion DSML available for allocation and no galaxy at a point in time should be allocated more than 100 zillion DSML or less than 0 zillion DSML. Galaxies with low existence expectancy index below 0.7 should be allocated at least 10% of the total energy available in the foreseeable future

3) Leaderboard is based on a combined scaled metric:

80% prediction task RMSE + 20% optimization task RMSE * lambda where lambda is a normalizing factor

4) Leaderboard is 80% public and 20% private

5) The submission should be in the following format:

Variable
Index
pred
out_pred

Description
Unique index from the test dataset in the ascending order
Prediction for the index on interest
Optimal energy allocation
sloy12

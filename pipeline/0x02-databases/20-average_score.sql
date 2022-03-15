-- script that creates a stored procedure ComputeAverageScoreForUser
-- that computes and store the average score for a student.

delimeter //
CREATE PROCEDURE ComputeAverageScoreForUser(IN new_user_id INT) BEGIN
UPDATE
        users
SET
        average_score = (
                SELECT
                        AVG(score)
                FROM
                        corrections
                WHERE
                        new_user_id = corrections.user_id
        )
WHERE
        id = new_user_id;

END //
delimeter;

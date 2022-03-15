-- script that creates a trigger that decreases the quantity of an item after adding a new order

DELIMITER || CREATE TRIGGER _UPDATE_ORDER_
AFTER
INSERT
        ON orders FOR EACH ROW BEGIN
UPDATE
        items
SET
        quantity = quantity - new.number
WHERE
        new.item_name = items.name;

END || DELIMITER;

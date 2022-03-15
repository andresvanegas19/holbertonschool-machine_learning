#!/bin/bash

$1 | docker exec -i db mysql -hlocalhost -uroot -p123 db_0

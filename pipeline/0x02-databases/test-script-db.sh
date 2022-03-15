#!/bin/bash

cat $1 | docker exec -i db mysql -hlocalhost -uroot -p123

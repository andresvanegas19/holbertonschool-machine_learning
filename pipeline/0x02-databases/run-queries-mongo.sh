#!/bin/bash

cat $1 | docker exec -i db mongo

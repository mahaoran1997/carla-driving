#!/bin/sh
for i in {2..14}; do sed 10s/.*/WeatherId=$i/ W1.ini > W$i.ini; done
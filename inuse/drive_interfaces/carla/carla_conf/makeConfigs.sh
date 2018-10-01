#!/bin/sh
for i in {2..14}; do sed 13s/.*/WeatherId=$i/ W1.ini > W$i.ini; done

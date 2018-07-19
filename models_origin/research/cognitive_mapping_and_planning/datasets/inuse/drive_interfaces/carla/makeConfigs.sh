#!/bin/sh
for i in {2..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1.ini > rcCarla_9Cams_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_pm5.ini > rcCarla_9Cams_pm5_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_pp5.ini > rcCarla_9Cams_pp5_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_w600.ini > rcCarla_9Cams_w600_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_w700.ini > rcCarla_9Cams_w700_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_w900.ini > rcCarla_9Cams_w900_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_w1000.ini > rcCarla_9Cams_w1000_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_w1100.ini > rcCarla_9Cams_w1100_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_w1200.ini > rcCarla_9Cams_w1200_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_z50.ini > rcCarla_9Cams_z50_W$i.ini; done
for i in {1..14}; do sed 10s/.*/WeatherId=$i/ rcCarla_9Cams_W1_z150.ini > rcCarla_9Cams_z150_W$i.ini; done

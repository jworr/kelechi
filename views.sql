
drop view price_lcu;

create view price_lcu as select * from producer_price where unit = 'LCU';

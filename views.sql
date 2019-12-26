
drop view price_lcu;
drop view price_lcu_common;

create view price_lcu as select * from producer_price where unit = 'LCU';

create view price_lcu_common as 
select * from price_lcu where item_code in (select item_code from price_lcu group by item_code having count(*) >= 5);

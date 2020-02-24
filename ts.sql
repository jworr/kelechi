
create view time_series as 
select
p.year,
p.name,
p.value as price,
p.unit as price_unit,
y.value as yield,
ha.value as area,
prod.value as production
from
price_lcu_common as p,
yield as y,
harvest_area as ha,
production as prod
where
p.item_code = y.item_code and
p.year = y.year and
y.item_code = ha.item_code and
y.year = ha.year and
ha.item_code = prod.item_code and
ha.year = prod.year

order by p.name, p.year;

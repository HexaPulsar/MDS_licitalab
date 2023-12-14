SELECT ao.category                     as AgileOrganismsCategory,
       au.region                       as AgileUnitsRegion,
       au.tax_number                   as OrganismoSolicitante,
       ab.code                         as AgileBuyingsCode,
       ab.description                  as AgileBuyingsDescription,
       ab.status                       as AgileBuyingsStatus,
       ab.currency                     as AgileBuyingsCurrency,
       available_amount                as AgileBuyingsAvailableAmount,
       ab.region                       as AgileBuyingsRegion,
       ai.name                         as AgileItemsName,
       ai.mp_id                        as AgileItemsMp_Id,
       ai.product_category             as AgileItemsProductCategory,
       ai.quantity                     as AgileItemsQuantity,
       ai.unit                         as AgileItemsUnit,
       "AgileOfferedItems".product     as AgileOfferedItemsProductoOfertado,
       "AgileOfferedItems".category    as AgileOfferedItemsCategoriaOfertada,
       "AgileOfferedItems".description as AgileOfferedItemsDescripcionOfertada,
       "AgileOfferedItems".quantity    as AgileOfferedItemsQuantityOfertada,
       "AgileOfferedItems".unit        as AgileOfferedItemsUnitOfertada,
       unit_price                      as AgileOfferedItemsPrecioUnitario,
       sub_total_price                 as AgileOfferedItemsPrecioTotal,
       p.tax_number                    as TaxNumberProvider,
       p.activity                      as ProviderActivity,
       p.region                        as ProviderRegion,
       p.city                          as ProviderCity,
       p.country                       as ProviderCountry,
       awarded                         as Adjudicada
FROM "AgileOfferedItems"
         JOIN "AgileItems" ai ON "AgileOfferedItems".agile_item_id = ai.id
         JOIN "Providers" p ON "AgileOfferedItems".provider_id = p.tax_number
         JOIN "AgileBuyings" ab ON ai.agbuy_id = ab.code
         JOIN "AgileUnits" au ON unit_id = au.tax_number
         JOIN "AgileOrganisms" ao ON organism_id = ao.id
WHERE ab."closing_date" >= '2023-04-01'
  AND ab."closing_date" <= '2023-05-31'
  AND ai.mp_id >= '42'
  AND ai.mp_id <= '60';
/** Carbon equivalents for the heat the leaks waste. Factors are stated in the UI so the numbers stay honest. */

export const CO2_KG_PER_KWH = 0.39; // US average grid intensity, electric heat (EIA 2023: ~0.86 lb CO2 per kWh)
export const TREE_KG_CO2_PER_YEAR = 22; // a mature tree absorbs roughly 48 lb of CO2 a year (USDA figure)
export const CAR_KG_CO2_PER_MILE = 0.4; // EPA average passenger car

export interface Carbon {
  kgPerYear: number;
  trees: number;
  milesDriven: number;
}

export function carbonFor(kwhPerYear: number): Carbon {
  const kgPerYear = kwhPerYear * CO2_KG_PER_KWH;
  return { kgPerYear, trees: kgPerYear / TREE_KG_CO2_PER_YEAR, milesDriven: kgPerYear / CAR_KG_CO2_PER_MILE };
}

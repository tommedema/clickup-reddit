export interface EpochRange {
  start: number;
  end: number;
}

export function toUnixSeconds(value: string | number): number {
  if (typeof value === 'number') {
    return Math.floor(value);
  }

  const asNumber = Number(value);
  if (!Number.isNaN(asNumber)) {
    return Math.floor(asNumber);
  }

  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    throw new Error(`Unable to parse time value: ${value}`);
  }

  return Math.floor(parsed / 1000);
}

export function epochSecondsNow(): number {
  return Math.floor(Date.now() / 1000);
}

export function epochToIso(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toISOString();
}

export function splitIntoMonthlyRanges(
  startEpoch: number,
  endEpoch: number,
  monthsPerBatch: number,
): EpochRange[] {
  if (monthsPerBatch <= 0) {
    throw new Error('monthsPerBatch must be greater than 0');
  }

  if (endEpoch <= startEpoch) {
    return [];
  }

  const ranges: EpochRange[] = [];
  const startDate = new Date(startEpoch * 1000);
  const endDate = new Date(endEpoch * 1000);

  let cursor = new Date(startDate);
  while (cursor < endDate) {
    const chunkStart = Math.floor(cursor.getTime() / 1000);
    const chunkEndDate = addMonths(cursor, monthsPerBatch);
    const boundedEnd = chunkEndDate > endDate ? endDate : chunkEndDate;
    const chunkEnd = Math.floor(boundedEnd.getTime() / 1000);

    if (chunkEnd <= chunkStart) {
      break;
    }

    ranges.push({ start: chunkStart, end: chunkEnd });
    cursor = new Date(boundedEnd);
  }

  const lastRange = ranges[ranges.length - 1];
  if (ranges.length === 0 || (lastRange && lastRange.end < endEpoch)) {
    const previousEnd = lastRange ? lastRange.end : startEpoch;
    if (endEpoch > previousEnd) {
      ranges.push({ start: previousEnd, end: endEpoch });
    }
  }

  return ranges;
}

function addMonths(date: Date, months: number): Date {
  const result = new Date(date);
  const targetMonth = result.getUTCMonth() + months;
  result.setUTCMonth(targetMonth);
  return result;
}

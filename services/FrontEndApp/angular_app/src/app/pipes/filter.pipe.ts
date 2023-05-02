// filter.pipe.ts

import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'appFilter' })
export class FilterPipe implements PipeTransform {
  /**
   * Transform
   *
   * @param {any[]} items
   * @param {string} searchText
   * @returns {any[]}
   */
  transform(items: any[], searchText: string): any[] {
    if (!items) {
      return [];
    }
    if (!searchText) {
      return items;
    }
    searchText = searchText.toLocaleLowerCase();

    return items.filter((it) => {
      return this.filter(it, searchText);
    });
  }

  filter(item: any, searchText: string): boolean {
    let result = false;
    if (typeof item == 'string') {
      result = item.toLocaleLowerCase().includes(searchText);
    } else if (typeof item == 'object') {
      Object.keys(item).forEach((key) => {
        if (this.filter(item[key], searchText)) {
          result = true;
        }
      });
    }

    return result;
  }
}

import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class NumberUtilsService {

  constructor() { }

  nFirstIntegers(n: number) {
    let res = new Array(n);
    for (let index = 0; index < res.length; index++) {
      res[index] = index;  
    }    
    return res;
  }
}
import { TestBed } from '@angular/core/testing';

import { NumberUtilsService } from './number-utils.service';

describe('NumberUtilsService', () => {
  let service: NumberUtilsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(NumberUtilsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

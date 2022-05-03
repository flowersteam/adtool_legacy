import { TestBed } from '@angular/core/testing';

import { PreparingLogService } from './preparing-log.service';

describe('PreparingLogService', () => {
  let service: PreparingLogService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PreparingLogService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

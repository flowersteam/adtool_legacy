import { TestBed } from '@angular/core/testing';

import { AppDbService } from './app-db.service';

describe('AppDbService', () => {
  let service: AppDbService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AppDbService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

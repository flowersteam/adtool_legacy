import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PreparingLogComponent } from './preparing-log.component';

describe('PreparingLogComponent', () => {
  let component: PreparingLogComponent;
  let fixture: ComponentFixture<PreparingLogComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PreparingLogComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(PreparingLogComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

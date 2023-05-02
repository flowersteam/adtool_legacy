import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CheckpointsDetailsComponent } from './checkpoints-details.component';

describe('CheckpointsDetailsComponent', () => {
  let component: CheckpointsDetailsComponent;
  let fixture: ComponentFixture<CheckpointsDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [CheckpointsDetailsComponent],
    }).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CheckpointsDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

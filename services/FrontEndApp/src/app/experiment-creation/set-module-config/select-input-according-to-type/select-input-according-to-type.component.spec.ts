import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SelectInputAccordingToTypeComponent } from './select-input-according-to-type.component';

describe('SelectInputAccordingToTypeComponent', () => {
  let component: SelectInputAccordingToTypeComponent;
  let fixture: ComponentFixture<SelectInputAccordingToTypeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SelectInputAccordingToTypeComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SelectInputAccordingToTypeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

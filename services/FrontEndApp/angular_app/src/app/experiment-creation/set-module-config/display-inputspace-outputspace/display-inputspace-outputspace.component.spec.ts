import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DisplayInputspaceOutputspaceComponent } from './display-inputspace-outputspace.component';

describe('DisplayInputspaceOutputspaceComponent', () => {
  let component: DisplayInputspaceOutputspaceComponent;
  let fixture: ComponentFixture<DisplayInputspaceOutputspaceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DisplayInputspaceOutputspaceComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DisplayInputspaceOutputspaceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

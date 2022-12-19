import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SetDiscoverySavingKeyComponent } from './set-discovery-saving-key.component';

describe('setDiscoverySavingKeyComponent', () => {
  let component: SetDiscoverySavingKeyComponent;
  let fixture: ComponentFixture<SetDiscoverySavingKeyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SetDiscoverySavingKeyComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SetDiscoverySavingKeyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

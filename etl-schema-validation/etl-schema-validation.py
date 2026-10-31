def validate_records(records: list, schema: list) -> list:
    """
    Returns a list of result dictionaries.
    """
    results = []
    
    for i, record in enumerate(records):
        errors = []
        for field in schema:
            col = field["column"]
            exp_type = field["type"]
            nullable = field.get("nullable", False)
            
            if col not in record:
                errors.append(f"{col}: missing")
                continue
            
            val = record[col]
            if val is None:
                if not nullable:
                    errors.append(f"{col}: null")
                continue
            
            actual_type = type(val)
            actual_type_name = actual_type.__name__
            is_valid_type = False
            
            if exp_type == "int":
                if actual_type is int:
                    is_valid_type = True
            elif exp_type == "float":
                if actual_type in (int, float):
                    is_valid_type = True
            elif exp_type == "str":
                if actual_type is str:
                    is_valid_type = True
                    
            if not is_valid_type:
                errors.append(f"{col}: expected {exp_type}, got {actual_type_name}")
                continue
                
            out_of_range = False
            if "min" in field and val < field["min"]:
                out_of_range = True
            if "max" in field and val > field["max"]:
                out_of_range = True
                
            if out_of_range:
                errors.append(f"{col}: out of range")
                continue
                
        results.append({
            "record_index": i,
            "is_valid": len(errors) == 0,
            "errors": errors
        })
        
    return results
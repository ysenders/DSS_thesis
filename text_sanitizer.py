
def sanitize_weka(text: str, remove_newlines=True, escape_doublequote=True, escape_singlequote=True,remove_separator=None) -> str:
    """
        sanitize this text for weka CSV importer. 

        Parameters:
            remove_newlines(Bool): removes newline charaters and replaces them with blank spaces. Default: True

            escape_doublequote(Bool): escapes a every doublequote character \\\" with \\\\\\\". Default: True.
                if False, it will remove the doublequote and replace it with empty String

            escape_singlequote(Bool): escapes a every singlequote character \\\' with \\\\\\\'. Default: True.
                if False, it will remove the singlequote and replace it with empty String

            remove_separator(str): removes the separator str passed as argument. Default: None
    """
    
    if remove_newlines:
        text = text.replace('\n', ' ') 

    if escape_doublequote:
        text = text.replace('"', '\\\"')
    else:
        text = text.replace('"', '')

    if escape_singlequote:
        text = text.replace("'", "\\\'") 
    else:
        text = text.replace("'", "")

    if remove_separator: 
        text = text.replace(remove_separator," ")

    return text
